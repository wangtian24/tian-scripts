import asyncio
import logging
import os
import time
import uuid
from decimal import Decimal

from fastapi import HTTPException
from sqlalchemy import select
from tenacity import retry, stop_after_attempt, wait_exponential
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack_with_user_name_bg
from ypl.backend.payment.coinbase.coinbase_payout import (
    CoinbaseRetailPayout,
    CoinbaseRetailPayoutError,
    TransactionStatus,
    get_all_transactions_from_coinbase,
    get_coinbase_retail_wallet_account_details,
    get_coinbase_retail_wallet_balance_for_currency,
    get_transaction_status,
    process_coinbase_retail_payout,
)
from ypl.backend.payment.facilitator import (
    BaseFacilitator,
    PaymentInstrumentError,
    PaymentProcessingError,
    PaymentResponse,
    PaymentStatusFetchError,
    PointTransactionCreationError,
    TransactionCreationError,
)
from ypl.backend.payment.payment import (
    CashoutPointTransactionRequest,
    PaymentTransactionRequest,
    create_cashout_point_transaction,
    create_payment_transaction,
    update_payment_transaction,
    update_user_points,
)
from ypl.backend.payment.payout_utils import (
    CASHOUT_TXN_COST,
    handle_failed_transaction,
)
from ypl.backend.payment.payout_utils import (
    get_destination_instrument_id as get_generic_destination_instrument_id,
)
from ypl.backend.payment.payout_utils import (
    get_source_instrument_id as get_generic_source_instrument_id,
)
from ypl.backend.utils.async_utils import create_background_task
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import (
    CurrencyEnum,
    PaymentInstrumentFacilitatorEnum,
    PaymentInstrumentIdentifierTypeEnum,
    PaymentTransaction,
    PaymentTransactionStatusEnum,
)
from ypl.db.point_transactions import PointsActionEnum

SYSTEM_USER_ID = "SYSTEM"
SLACK_WEBHOOK_CASHOUT = os.getenv("SLACK_WEBHOOK_CASHOUT")
RETRY_ATTEMPTS = 3
RETRY_WAIT_MULTIPLIER = 1
RETRY_WAIT_MIN = 4
RETRY_WAIT_MAX = 15


class CoinbaseFacilitator(BaseFacilitator):
    def __init__(
        self,
        currency: CurrencyEnum,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        facilitator: PaymentInstrumentFacilitatorEnum,
    ):
        super().__init__(currency, destination_identifier_type, facilitator)

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    )
    async def get_source_instrument_id(self) -> uuid.UUID:
        return await get_generic_source_instrument_id(
            PaymentInstrumentFacilitatorEnum.COINBASE,
            PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
        )

    async def get_destination_instrument_id(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        instrument_metadata: dict | None = None,
    ) -> uuid.UUID:
        return await get_generic_destination_instrument_id(
            PaymentInstrumentFacilitatorEnum.COINBASE,
            user_id,
            destination_identifier,
            destination_identifier_type,
            instrument_metadata,
        )

    async def get_balance(self, currency: CurrencyEnum, payment_transaction_id: uuid.UUID | None = None) -> Decimal:
        """Get the balance for a specific currency.

        Args:
            currency: The currency to get balance for
            payment_transaction_id: The ID of the payment transaction, if this request is part of a payment transaction.
        Returns:
            Decimal: The balance amount
        """
        coinbase_balance = await get_coinbase_retail_wallet_balance_for_currency(currency)
        balance = coinbase_balance["balance"]
        if isinstance(balance, str):
            return Decimal(balance)
        return balance

    async def _send_payment_request(
        self,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
        usd_amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        destination_additional_details: dict | None = None,
        *,
        # TODO: Make this a standard parameter.
        payment_transaction_id: uuid.UUID | None = None,
    ) -> PaymentResponse:
        start_time = time.time()
        # coinbase requires a minimum of 1 USD
        if usd_amount < 1:
            log_dict = {
                "message": "Minimum amount of 1 USD is required",
                "user_id": user_id,
                "amount_requested": str(usd_amount),
                "credits_to_cashout": str(credits_to_cashout),
            }
            logging.warning(json_dumps(log_dict))
            # UI may not show this error message, as there's going to be a UI validation too before this.
            raise ValueError("You need to cash out at least 1 USD worth of credits")
        try:
            try:
                credits_to_cashout += CASHOUT_TXN_COST
                # Get the balance of the source instrument
                source_instrument_balance = await self.get_balance(self.currency, payment_transaction_id)

                if source_instrument_balance < amount:
                    log_dict = {
                        "message": "Source instrument does not have enough balance",
                        "user_id": user_id,
                        "amount": str(amount),
                        "source_instrument_balance": str(source_instrument_balance),
                    }
                    logging.error(json_dumps(log_dict))
                    raise PaymentInstrumentError("Source instrument does not have enough balance")

                source_instrument_id = await self.get_source_instrument_id()
                destination_instrument_id = await self.get_destination_instrument_id(
                    user_id, destination_identifier, destination_identifier_type
                )
            except HTTPException as e:
                raise e
            except Exception as e:
                log_dict = {
                    "message": "Failed to get payment instruments",
                    "user_id": user_id,
                    "error": str(e),
                    "destination_identifier": destination_identifier,
                }
                logging.exception(json_dumps(log_dict))
                raise PaymentInstrumentError("Failed to get payment instruments") from e

            try:
                payment_transaction_request = PaymentTransactionRequest(
                    currency=self.currency,
                    amount=amount,
                    usd_amount=usd_amount,
                    source_instrument_id=source_instrument_id,
                    destination_instrument_id=destination_instrument_id,
                    status=PaymentTransactionStatusEnum.NOT_STARTED,
                    additional_info={
                        "user_id": user_id,
                        "destination_identifier": destination_identifier,
                        "destination_identifier_type": str(destination_identifier_type),
                    },
                )
                payment_transaction_id = await create_payment_transaction(payment_transaction_request)
            except Exception as e:
                log_dict = {
                    "message": "Failed to create payment transaction",
                    "user_id": user_id,
                    "amount": str(amount),
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                raise TransactionCreationError("Failed to create payment transaction") from e

            try:
                point_transaction_request = CashoutPointTransactionRequest(
                    user_id=user_id,
                    point_delta=-credits_to_cashout,
                    action_type=PointsActionEnum.CASHOUT,
                    cashout_payment_transaction_id=payment_transaction_id,
                )
                point_transaction_id = await create_cashout_point_transaction(point_transaction_request)
            except Exception as e:
                log_dict = {
                    "message": "Failed to create point transaction",
                    "user_id": user_id,
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                await handle_failed_transaction(
                    payment_transaction_id,
                    None,
                    user_id,
                    credits_to_cashout,
                    amount,
                    usd_amount,
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=False,
                    currency=self.currency,
                )
                raise PointTransactionCreationError("Failed to create point transaction") from e

            try:
                await update_user_points(user_id, -credits_to_cashout)
            except Exception as e:
                log_dict = {
                    "message": "Failed to update user points",
                    "user_id": user_id,
                    "payment_transaction_id": str(payment_transaction_id),
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                await handle_failed_transaction(
                    payment_transaction_id,
                    point_transaction_id,
                    user_id,
                    credits_to_cashout,
                    amount,
                    usd_amount,
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=False,
                    currency=self.currency,
                )
                raise PaymentProcessingError("Failed to update user points") from e

            try:
                coinbase_payout = CoinbaseRetailPayout(
                    user_id=user_id,
                    amount=amount,
                    to_address=destination_identifier,
                    currency=self.currency,
                    payment_transaction_id=payment_transaction_id,
                )
                account_id, transaction_id, transaction_status = await process_coinbase_retail_payout(coinbase_payout)

                await update_payment_transaction(
                    payment_transaction_id,
                    partner_reference_id=transaction_id,
                    status=self._map_coinbase_status_to_internal(transaction_status),
                    customer_reference_id=transaction_id,
                )

                if transaction_status != TransactionStatus.COMPLETED.value:
                    # Start monitoring in background task
                    create_background_task(
                        self._monitor_transaction_completion(
                            account_id=account_id,
                            transaction_id=transaction_id,
                            payment_transaction_id=payment_transaction_id,
                            points_transaction_id=point_transaction_id,
                            user_id=user_id,
                            credits_to_cashout=credits_to_cashout,
                            amount=amount,
                            usd_amount=usd_amount,
                            source_instrument_id=source_instrument_id,
                            destination_instrument_id=destination_instrument_id,
                            destination_identifier=destination_identifier,
                            destination_identifier_type=destination_identifier_type,
                        )
                    )

                # Log success
                end_time = time.time()
                log_dict = {
                    "message": ":white_check_mark: Success - Coinbase retail payout submitted",
                    "duration": str(end_time - start_time),
                    "user_id": user_id,
                    "amount": str(amount),
                    "usd_amount": str(usd_amount),
                    "credits_to_cashout": str(credits_to_cashout),
                    "source_instrument_id": str(source_instrument_id),
                    "destination_instrument_id": str(destination_instrument_id),
                    "destination_identifier": destination_identifier,
                    "currency": self.currency.value,
                    "account_id": account_id,
                    "transaction_id": transaction_id,
                    "payment_transaction_id": str(payment_transaction_id),
                    "points_transaction_id": str(point_transaction_id),
                }
                logging.info(json_dumps(log_dict))
                post_to_slack_with_user_name_bg(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT)
                return PaymentResponse(
                    payment_transaction_id=payment_transaction_id,
                    transaction_status=PaymentTransactionStatusEnum.PENDING,
                    customer_reference_id=transaction_id,
                )

            except CoinbaseRetailPayoutError as e:
                log_dict = {
                    "message": "Failed to process Coinbase retail payout",
                    "user_id": user_id,
                    "amount": str(amount),
                    "destination_identifier": destination_identifier,
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                await handle_failed_transaction(
                    payment_transaction_id,
                    point_transaction_id,
                    user_id,
                    credits_to_cashout,
                    amount,
                    usd_amount,
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=True,
                    currency=self.currency,
                )
                raise ValueError(str(e)) from e
            except Exception as e:
                log_dict = {
                    "message": "Failed to process Coinbase retail payout. Reversing transaction.",
                    "user_id": user_id,
                    "amount": str(amount),
                    "destination_identifier": destination_identifier,
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                await handle_failed_transaction(
                    payment_transaction_id,
                    point_transaction_id,
                    user_id,
                    credits_to_cashout,
                    amount,
                    usd_amount,
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=True,
                    currency=self.currency,
                )
                raise PaymentProcessingError("Failed to process Coinbase retail payout") from e

        except HTTPException as e:
            raise e
        except Exception as e:
            log_dict = {
                "message": "Unexpected error in Coinbase retail payout processing",
                "user_id": user_id,
                "amount": str(amount),
                "destination_identifier": destination_identifier,
                "error": str(e),
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentProcessingError("Coinbase retail payout processing failed") from e

    @staticmethod
    def _map_coinbase_status_to_internal(status: str) -> PaymentTransactionStatusEnum:
        """Map Coinbase's transaction status to our internal PaymentTransactionStatusEnum.

        Args:
            status: The Coinbase transaction status

        Returns:
            PaymentTransactionStatusEnum: The corresponding internal status
        """
        if status == TransactionStatus.COMPLETED.value:
            return PaymentTransactionStatusEnum.SUCCESS
        elif status == TransactionStatus.FAILED.value:
            return PaymentTransactionStatusEnum.FAILED
        elif status == TransactionStatus.PENDING.value:
            return PaymentTransactionStatusEnum.PENDING
        else:  # TransactionStatus.UNKNOWN.value
            # For unknown status, keep it as PENDING since we don't know if it failed
            return PaymentTransactionStatusEnum.PENDING

    async def get_payment_status(self, payment_transaction_id: uuid.UUID) -> PaymentResponse:
        log_dict = {
            "message": "Getting Coinbase retail payout status",
            "payment_transaction_id": str(payment_transaction_id),
            "currency": self.currency.value,
        }
        async with get_async_session() as session:
            query = select(PaymentTransaction).where(
                PaymentTransaction.payment_transaction_id == payment_transaction_id  # type: ignore
            )
            result = await session.execute(query)
            transaction = result.scalar_one_or_none()
            if not transaction:
                raise PaymentStatusFetchError(f"Payment transaction {payment_transaction_id} not found")

            if transaction.status in (
                PaymentTransactionStatusEnum.SUCCESS,
                PaymentTransactionStatusEnum.FAILED,
                PaymentTransactionStatusEnum.REVERSED,
            ):
                return PaymentResponse(
                    payment_transaction_id=payment_transaction_id,
                    transaction_status=transaction.status,
                    customer_reference_id=str(transaction.partner_reference_id),
                    partner_reference_id=str(transaction.partner_reference_id),
                )
        partner_reference_id = transaction.partner_reference_id
        if not partner_reference_id:
            all_transactions = await get_all_transactions_from_coinbase()
            for each_transaction in all_transactions["data"]:
                if each_transaction["idem"] == transaction.payment_transaction_id:
                    partner_reference_id = each_transaction["id"]
                    break

        if not partner_reference_id:
            raise PaymentStatusFetchError("Partner reference ID not found")

        partner_reference_id = str(partner_reference_id)
        accounts = await get_coinbase_retail_wallet_account_details()
        account_info = accounts.get(self.currency.value, {})
        account_id = account_info.get("account_id")
        if not account_id:
            raise PaymentStatusFetchError("Account ID not found")

        status = await get_transaction_status(str(account_id), partner_reference_id)
        log_dict = {
            "message": "Coinbase retail payout status",
            "payment_transaction_id": str(payment_transaction_id),
            "status": status,
        }
        logging.info(json_dumps(log_dict))
        return PaymentResponse(
            payment_transaction_id=payment_transaction_id,
            transaction_status=self._map_coinbase_status_to_internal(status),
            customer_reference_id=partner_reference_id,
        )

    async def _monitor_transaction_completion(
        self,
        account_id: str,
        transaction_id: str,
        payment_transaction_id: uuid.UUID,
        points_transaction_id: uuid.UUID,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
        usd_amount: Decimal,
        source_instrument_id: uuid.UUID,
        destination_instrument_id: uuid.UUID,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
    ) -> None:
        """Monitor Coinbase retail payout completion and handle success/failure.

        Args:
            account_id: The Coinbase account ID
            transaction_id: The Coinbase transaction ID to monitor
            payment_transaction_id: The ID of the payment transaction
            points_transaction_id: The ID of the points transaction
            user_id: The user ID
            credits_to_cashout: The number of credits being cashed out
            amount: The amount being transferred
            source_instrument_id: The source payment instrument ID
            destination_instrument_id: The destination payment instrument ID
            destination_identifier: The destination wallet address
            destination_identifier_type: The type of destination identifier
        """
        try:
            start_time = time.time()
            max_wait_time, poll_interval = self.get_polling_config()

            while (time.time() - start_time) < max_wait_time:
                status = await get_transaction_status(account_id, transaction_id)
                if status == TransactionStatus.COMPLETED.value:
                    await update_payment_transaction(
                        payment_transaction_id,
                        partner_reference_id=transaction_id,
                        status=PaymentTransactionStatusEnum.SUCCESS,
                        customer_reference_id=transaction_id,
                    )
                    log_dict = {
                        "message": "Coinbase retail payout completed",
                        "user_id": user_id,
                        "transaction_id": transaction_id,
                        "status": status,
                        "elapsed_time": time.time() - start_time,
                    }
                    logging.info(json_dumps(log_dict))
                    return
                elif status == TransactionStatus.FAILED.value:
                    log_dict = {
                        "message": "Coinbase retail payout failed",
                        "user_id": user_id,
                        "transaction_id": transaction_id,
                        "status": status,
                        "elapsed_time": time.time() - start_time,
                    }
                    logging.error(json_dumps(log_dict))

                    # Handle the failed transaction
                    await handle_failed_transaction(
                        payment_transaction_id=payment_transaction_id,
                        points_transaction_id=points_transaction_id,
                        user_id=user_id,
                        credits_to_cashout=credits_to_cashout,
                        amount=amount,
                        usd_amount=usd_amount,
                        source_instrument_id=source_instrument_id,
                        destination_instrument_id=destination_instrument_id,
                        destination_identifier=destination_identifier,
                        destination_identifier_type=destination_identifier_type,
                        update_points=True,
                        currency=self.currency,
                    )
                    return

                await asyncio.sleep(poll_interval)

            # If we get here, we've timed out
            # Do not reverse the transaction here as the txn might still complete
            log_dict = {
                "message": ":x: Failure - Coinbase retail payout monitoring timed out\n"
                f"account_id: {account_id}\n"
                f"transaction_id: {transaction_id}\n"
                f"payment_transaction_id: {payment_transaction_id}\n"
                f"points_transaction_id: {points_transaction_id}\n"
                f"user_id: {user_id}\n"
                f"credits_to_cashout: {credits_to_cashout}\n"
                f"amount: {amount}\n"
                f"usd_amount: {usd_amount}\n"
                f"source_instrument_id: {source_instrument_id}\n"
                f"destination_instrument_id: {destination_instrument_id}\n"
                f"destination_identifier: {destination_identifier}\n"
                f"destination_identifier_type: {destination_identifier_type}\n"
                f"status: {status}\n",
            }
            logging.error(json_dumps(log_dict))

            # TODO: Send alert to Slack
            post_to_slack_with_user_name_bg(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT)

        except Exception as e:
            log_dict = {
                "message": ":x: Failure - Error monitoring Coinbase retail payout completion",
                "user_id": user_id,
                "transaction_id": transaction_id,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
            }
            logging.error(json_dumps(log_dict))
            raise PaymentProcessingError("Failed to monitor Coinbase retail payout completion") from e

    @staticmethod
    def get_polling_config() -> tuple[int, int]:
        """
        Returns polling configuration for Coinbase transactions.
        Returns:
            tuple: (max_wait_time_seconds, poll_interval_seconds)
        """
        return (
            5 * 60,  # 5 minutes in seconds
            10,  # 10 seconds
        )
