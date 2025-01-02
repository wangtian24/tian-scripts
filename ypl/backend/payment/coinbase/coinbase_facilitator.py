import asyncio
import logging
import os
import time
import uuid
from decimal import Decimal

from sqlalchemy import func
from sqlmodel import select
from tenacity import retry, stop_after_attempt, wait_exponential
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.payment.coinbase.coinbase_payout import (
    CoinbaseRetailPayout,
    TransactionStatus,
    get_coinbase_retail_wallet_balance_for_currency,
    get_transaction_status,
    process_coinbase_retail_payout,
)
from ypl.backend.payment.facilitator import (
    BaseFacilitator,
    PaymentInstrumentError,
    PaymentInstrumentNotFoundError,
    PaymentProcessingError,
    PaymentResponse,
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
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import (
    CurrencyEnum,
    PaymentInstrument,
    PaymentInstrumentFacilitatorEnum,
    PaymentInstrumentIdentifierTypeEnum,
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
        async with get_async_session() as session:
            query = select(PaymentInstrument).where(
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.COINBASE,
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
                PaymentInstrument.user_id == SYSTEM_USER_ID,
                PaymentInstrument.deleted_at.is_(None),  # type: ignore
            )

            result = await session.exec(query)
            instrument = result.first()
            if not instrument:
                log_dict = {
                    "message": "Source payment instrument not found",
                    "identifier_type": PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
                    "facilitator": PaymentInstrumentFacilitatorEnum.COINBASE,
                    "user_id": SYSTEM_USER_ID,
                }
                logging.exception(json_dumps(log_dict))
                raise PaymentInstrumentNotFoundError(
                    f"Payment instrument not found for {PaymentInstrumentFacilitatorEnum.COINBASE}"
                )
            return instrument.payment_instrument_id

    async def get_destination_instrument_id(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
    ) -> uuid.UUID:
        async with get_async_session() as session:
            query = select(PaymentInstrument).where(
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.COINBASE,
                PaymentInstrument.identifier_type == destination_identifier_type,
                func.lower(PaymentInstrument.identifier) == destination_identifier.lower(),
                PaymentInstrument.user_id == user_id,
                PaymentInstrument.deleted_at.is_(None),  # type: ignore
            )
            result = await session.exec(query)
            instrument = result.first()
            if not instrument:
                log_dict = {
                    "message": "Destination payment instrument not found. Creating a new one.",
                    "identifier_type": destination_identifier_type,
                    "facilitator": PaymentInstrumentFacilitatorEnum.COINBASE,
                    "user_id": user_id,
                    "identifier": destination_identifier,
                }
                logging.info(json_dumps(log_dict))
                instrument = PaymentInstrument(
                    facilitator=PaymentInstrumentFacilitatorEnum.COINBASE,
                    identifier_type=destination_identifier_type,
                    identifier=destination_identifier,
                    user_id=user_id,
                )
                session.add(instrument)
                await session.commit()
                return instrument.payment_instrument_id
            return instrument.payment_instrument_id

    async def get_balance(self, currency: CurrencyEnum) -> Decimal:
        """Get the balance for a specific currency.

        Args:
            currency: The currency to get balance for

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
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        destination_additional_details: dict | None = None,
    ) -> PaymentResponse:
        start_time = time.time()
        try:
            try:
                # Get the balance of the source instrument
                source_instrument_balance = await self.get_balance(self.currency)

                if source_instrument_balance < amount:
                    log_dict = {
                        "message": "Source instrument does not have enough balance",
                        "user_id": user_id,
                        "amount": str(amount),
                        "source_instrument_balance": str(source_instrument_balance),
                    }
                    logging.error(json_dumps(log_dict))
                    raise ValueError("Source instrument does not have enough balance")

                source_instrument_id = await self.get_source_instrument_id()
                destination_instrument_id = await self.get_destination_instrument_id(
                    user_id, destination_identifier, destination_identifier_type
                )
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
                    source_instrument_id=source_instrument_id,
                    destination_instrument_id=destination_instrument_id,
                    status=PaymentTransactionStatusEnum.NOT_STARTED,
                    additional_info={
                        "user_id": user_id,
                        "destination_identifier": destination_identifier,
                        "destination_identifier_type": destination_identifier_type,
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
                await self._handle_failed_transaction(
                    payment_transaction_id,
                    None,
                    user_id,
                    credits_to_cashout,
                    amount,
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=False,
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
                await self._handle_failed_transaction(
                    payment_transaction_id,
                    point_transaction_id,
                    user_id,
                    credits_to_cashout,
                    amount,
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=False,
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
                )

                # Start monitoring in background task
                asyncio.create_task(
                    self._monitor_transaction_completion(
                        account_id=account_id,
                        transaction_id=transaction_id,
                        payment_transaction_id=payment_transaction_id,
                        points_transaction_id=point_transaction_id,
                        user_id=user_id,
                        credits_to_cashout=credits_to_cashout,
                        amount=amount,
                        source_instrument_id=source_instrument_id,
                        destination_instrument_id=destination_instrument_id,
                        destination_identifier=destination_identifier,
                        destination_identifier_type=destination_identifier_type,
                    )
                )

                # Log success
                end_time = time.time()
                log_dict = {
                    "message": "Successfully submitted Coinbase retail payout",
                    "duration": str(end_time - start_time),
                    "user_id": user_id,
                    "amount": str(amount),
                    "credits_to_cashout": str(credits_to_cashout),
                    "source_instrument_id": str(source_instrument_id),
                    "destination_instrument_id": str(destination_instrument_id),
                    "destination_identifier": destination_identifier,
                    "currency": self.currency.value,
                    "transaction_id": transaction_id,
                }
                logging.info(json_dumps(log_dict))
                asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
                return PaymentResponse(
                    payment_transaction_id=payment_transaction_id,
                    transaction_status=PaymentTransactionStatusEnum.PENDING,
                    customer_reference_id=transaction_id,
                )

            except Exception as e:
                log_dict = {
                    "message": "Failed to process Coinbase retail payout. Reversing transaction.",
                    "user_id": user_id,
                    "amount": str(amount),
                    "destination_identifier": destination_identifier,
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                await self._handle_failed_transaction(
                    payment_transaction_id,
                    point_transaction_id,
                    user_id,
                    credits_to_cashout,
                    amount,
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=True,
                )
                raise PaymentProcessingError("Failed to process Coinbase retail payout") from e

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

    async def _handle_failed_transaction(
        self,
        payment_transaction_id: uuid.UUID,
        points_transaction_id: uuid.UUID | None,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
        source_instrument_id: uuid.UUID,
        destination_instrument_id: uuid.UUID,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        update_points: bool,
    ) -> None:
        """Handle cleanup for failed transactions"""
        try:
            log_dict = {
                "message": "Failed to process Coinbase retail payout. Reversing transaction.",
                "user_id": user_id,
                "payment_transaction_id": str(payment_transaction_id),
                "points_transaction_id": str(points_transaction_id),
                "credits_to_cashout": str(credits_to_cashout),
                "amount": str(amount),
                "source_instrument_id": str(source_instrument_id),
                "destination_instrument_id": str(destination_instrument_id),
                "destination_identifier": destination_identifier,
                "destination_identifier_type": destination_identifier_type,
            }
            logging.info(json_dumps(log_dict))
            asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

            await update_payment_transaction(payment_transaction_id, status=PaymentTransactionStatusEnum.FAILED)
            if update_points:
                await update_user_points(user_id, credits_to_cashout)

            reversal_request = PaymentTransactionRequest(
                currency=self.currency,
                amount=amount,
                source_instrument_id=source_instrument_id,
                destination_instrument_id=destination_instrument_id,
                status=PaymentTransactionStatusEnum.REVERSED,
                additional_info={
                    "user_id": user_id,
                    "destination_identifier": destination_identifier,
                    "destination_identifier_type": destination_identifier_type,
                    "reversal_transaction_id": payment_transaction_id,
                },
            )
            payment_transaction_id = await create_payment_transaction(reversal_request)
            if points_transaction_id:
                points_transaction_id = await create_cashout_point_transaction(
                    CashoutPointTransactionRequest(
                        user_id=user_id,
                        point_delta=credits_to_cashout,
                        action_type=PointsActionEnum.CASHOUT_REVERSED,
                        cashout_payment_transaction_id=payment_transaction_id,
                    )
                )
            log_dict = {
                "message": "Successfully reversed Coinbase retail payout",
                "payment_transaction_id": str(payment_transaction_id),
                "points_transaction_id": str(points_transaction_id),
                "user_id": user_id,
                "amount": str(amount),
                "source_instrument_id": str(source_instrument_id),
                "destination_instrument_id": str(destination_instrument_id),
                "destination_identifier": destination_identifier,
                "destination_identifier_type": destination_identifier_type,
            }
            logging.info(json_dumps(log_dict))
            asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
        except Exception as e:
            error_message = str(e)
            log_dict = {
                "message": "Failed to handle failed Coinbase retail payout cleanup",
                "payment_transaction_id": str(payment_transaction_id),
                "points_transaction_id": str(points_transaction_id),
                "user_id": user_id,
                "amount": str(amount),
                "source_instrument_id": str(source_instrument_id),
                "destination_instrument_id": str(destination_instrument_id),
                "destination_identifier": destination_identifier,
                "destination_identifier_type": destination_identifier_type,
                "error": error_message,
            }
            logging.exception(json_dumps(log_dict))
            asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

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

    async def get_payment_status(self, payment_reference_id: str) -> PaymentTransactionStatusEnum:
        # TODO: Implement this as an account_id is required to get the status
        return PaymentTransactionStatusEnum.PENDING

    async def _monitor_transaction_completion(
        self,
        account_id: str,
        transaction_id: str,
        payment_transaction_id: uuid.UUID,
        points_transaction_id: uuid.UUID,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
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
                    await self._handle_failed_transaction(
                        payment_transaction_id=payment_transaction_id,
                        points_transaction_id=points_transaction_id,
                        user_id=user_id,
                        credits_to_cashout=credits_to_cashout,
                        amount=amount,
                        source_instrument_id=source_instrument_id,
                        destination_instrument_id=destination_instrument_id,
                        destination_identifier=destination_identifier,
                        destination_identifier_type=destination_identifier_type,
                        update_points=True,
                    )
                    return

                await asyncio.sleep(poll_interval)

            # If we get here, we've timed out
            log_dict = {
                "message": "Coinbase retail payout monitoring timed out",
                "user_id": user_id,
                "transaction_id": transaction_id,
                "timeout": True,
                "elapsed_time": time.time() - start_time,
            }
            logging.error(json_dumps(log_dict))

            # Handle the failed transaction
            await self._handle_failed_transaction(
                payment_transaction_id=payment_transaction_id,
                points_transaction_id=points_transaction_id,
                user_id=user_id,
                credits_to_cashout=credits_to_cashout,
                amount=amount,
                source_instrument_id=source_instrument_id,
                destination_instrument_id=destination_instrument_id,
                destination_identifier=destination_identifier,
                destination_identifier_type=destination_identifier_type,
                update_points=True,
            )

        except Exception as e:
            log_dict = {
                "message": "Error monitoring Coinbase retail payout completion",
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
