import asyncio
import logging
import os
import time
import uuid
from decimal import Decimal
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack_with_user_name
from ypl.backend.payment.checkout_com.checkout_com_payout import (
    AccountHolder,
    BankAccountInstrument,
    BillingAddress,
    BillingDescriptor,
    CheckoutPayout,
    CheckoutPayoutError,
    CurrencyAccountSource,
    IdDestination,
    Instruction,
    TransactionStatus,
    create_checkout_instrument,
    get_source_instrument_balance,
    get_transaction_status,
    process_checkout_payout,
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
    get_destination_instrument_id,
    get_instrument_details,
    handle_failed_transaction,
)
from ypl.backend.payment.payout_utils import (
    get_source_instrument_id as get_generic_source_instrument_id,
)
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import (
    CurrencyEnum,
    PaymentInstrument,
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


class CheckoutFacilitator(BaseFacilitator):
    def __init__(
        self,
        currency: CurrencyEnum,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        facilitator: PaymentInstrumentFacilitatorEnum,
        destination_additional_details: dict | None = None,
    ):
        super().__init__(currency, destination_identifier_type, facilitator)

    async def get_source_instrument_id(self) -> uuid.UUID:
        return await get_generic_source_instrument_id(
            PaymentInstrumentFacilitatorEnum.CHECKOUT_COM,
            PaymentInstrumentIdentifierTypeEnum.PARTNER_IDENTIFIER,
        )

    async def get_destination_instrument_id(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        destination_additional_details: dict | None = None,
    ) -> uuid.UUID:
        # For checkout.com, we need to create an instrument on the checkout.com side
        # and then use that instrument ID here.
        # Right now, we would support only 1 instrument per user. So check if the
        # instrument exists and if not, create it.
        async with get_async_session() as session:
            query = select(PaymentInstrument).where(
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.CHECKOUT_COM,  # type: ignore
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.PARTNER_IDENTIFIER,  # type: ignore
                PaymentInstrument.user_id == user_id,  # type: ignore
                PaymentInstrument.deleted_at.is_(None),  # type: ignore
            )
            result = await session.execute(query)
            existing_instruments = result.scalar_one_or_none()

        if not existing_instruments:
            # Create the instrument on the checkout.com side
            if not destination_additional_details:
                raise PaymentInstrumentError("Missing required bank account details")

            billing_address = BillingAddress(
                address_line1=destination_additional_details["address_line1"],
                city=destination_additional_details["city"],
                state=destination_additional_details["state"],
                zip=destination_additional_details["zip"],
                country=destination_additional_details["country"],
            )
            account_holder = AccountHolder(
                type="individual",
                first_name=destination_additional_details["first_name"],
                last_name=destination_additional_details["last_name"],
                billing_address=billing_address,
            )
            instrument = BankAccountInstrument(
                type="bank_account",
                currency="USD",
                account_type="current",
                account_number=destination_additional_details["account_number"],
                bank_code=destination_additional_details["bank_code"],
                country=destination_additional_details["country"],
                account_holder=account_holder,
            )
            instrument_id = await create_checkout_instrument(user_id, instrument)

            return await get_destination_instrument_id(
                PaymentInstrumentFacilitatorEnum.CHECKOUT_COM,
                user_id,
                instrument_id,
                PaymentInstrumentIdentifierTypeEnum.PARTNER_IDENTIFIER,
                None,
            )

        return UUID(str(existing_instruments.payment_instrument_id))

    async def get_balance(self, currency: CurrencyEnum, payment_transaction_id: UUID | None = None) -> Decimal:
        """Get the balance for a specific currency.

        Args:
            currency: The currency to get balance for
            payment_transaction_id: The ID of the payment transaction, if this request is part of a payment transaction.
        Returns:
            Decimal: The balance amount
        """
        return await get_source_instrument_balance()

    async def _send_payment_request(
        self,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        destination_additional_details: dict | None = None,
        payment_transaction_id: uuid.UUID | None = None,
    ) -> PaymentResponse:
        start_time = time.time()

        try:
            try:
                credits_to_cashout += CASHOUT_TXN_COST
                # TODO verify this check before reenabling it
                # source_instrument_balance = await self.get_balance(self.currency, payment_transaction_id)

                # if source_instrument_balance < amount:
                #     log_dict = {
                #         "message": "Source instrument does not have enough balance",
                #         "user_id": user_id,
                #         "amount": str(amount),
                #         "source_instrument_balance": str(source_instrument_balance),
                #     }
                #     logging.error(json_dumps(log_dict))
                #     raise PaymentInstrumentError("Source instrument does not have enough balance")

                source_instrument_id = await self.get_source_instrument_id()
                destination_instrument_id = await self.get_destination_instrument_id(
                    user_id, destination_identifier, destination_identifier_type, destination_additional_details
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

            #  we need to get the account details for the source and destination instruments
            #  and pass it to the payout request for checkout.com
            source_instrument = await get_instrument_details(source_instrument_id)
            destination_instrument = await get_instrument_details(destination_instrument_id)

            source_instrument_identifier = source_instrument.identifier if source_instrument else None
            destination_instrument_identifier = destination_instrument.identifier if destination_instrument else None

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
                        "destination_identifier_type": str(destination_identifier_type),
                    },
                )
                payment_transaction_id = await create_payment_transaction(payment_transaction_request)
            except Exception as e:
                log_dict = {
                    "message": "Checkout.com: Failed to create payment transaction",
                    "user_id": user_id,
                    "amount": str(amount),
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                raise TransactionCreationError("Checkout.com: Failed to create payment transaction") from e

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
                    "message": "Checkout.com: Failed to create point transaction",
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
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=False,
                    currency=self.currency,
                )
                raise PointTransactionCreationError("Checkout.com: Failed to create point transaction") from e

            try:
                await update_user_points(user_id, -credits_to_cashout)
            except Exception as e:
                log_dict = {
                    "message": "Checkout.com: Failed to update user points",
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
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=False,
                    currency=self.currency,
                )
                raise PaymentProcessingError("Checkout.com: Failed to update user points") from e

            try:
                if not source_instrument_identifier or not destination_instrument_identifier:
                    raise PaymentInstrumentError("Checkout.com: Source or destination instrument not found")

                checkout_payout = CheckoutPayout(
                    source=CurrencyAccountSource(
                        type="currency_account",
                        id=source_instrument_identifier,
                    ),
                    destination=IdDestination(
                        type="id",
                        id=destination_instrument_identifier,
                    ),
                    amount=amount,
                    currency=self.currency,
                    reference=str(payment_transaction_id),
                    billing_descriptor=BillingDescriptor(
                        reference="YUPP Payout",
                    ),
                    instruction=Instruction(
                        purpose="YUPP Payout",
                    ),
                )
                transaction_token, transaction_status = await process_checkout_payout(checkout_payout)

                await update_payment_transaction(
                    payment_transaction_id,
                    partner_reference_id=transaction_token,
                    status=self._map_checkout_status_to_internal(transaction_status),
                    customer_reference_id=transaction_token,
                )

                # checkout.com mostly returns pending status
                if transaction_status not in (TransactionStatus.PAID.value, TransactionStatus.DECLINED.value):
                    # Start monitoring in background task
                    asyncio.create_task(
                        self._monitor_transaction_completion(
                            transaction_token=transaction_token,
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
                    "message": ":white_check_mark: Success - Checkout.com payout submitted",
                    "duration": str(end_time - start_time),
                    "user_id": user_id,
                    "amount": str(amount),
                    "credits_to_cashout": str(credits_to_cashout),
                    "source_instrument_id": str(source_instrument_id),
                    "destination_instrument_id": str(destination_instrument_id),
                    "destination_identifier": destination_identifier,
                    "currency": self.currency.value,
                    "transaction_token": transaction_token,
                    "payment_transaction_id": str(payment_transaction_id),
                    "points_transaction_id": str(point_transaction_id),
                }
                logging.info(json_dumps(log_dict))
                asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
                return PaymentResponse(
                    payment_transaction_id=payment_transaction_id,
                    transaction_status=PaymentTransactionStatusEnum.PENDING,
                    customer_reference_id=transaction_token,
                )

            except CheckoutPayoutError as e:
                log_dict = {
                    "message": "Checkout.com: Failed to process Checkout.com payout",
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
                    "message": "Checkout.com: Failed to process Checkout.com payout. Reversing transaction.",
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
                    source_instrument_id,
                    destination_instrument_id,
                    destination_identifier,
                    destination_identifier_type,
                    update_points=True,
                    currency=self.currency,
                )
                raise PaymentProcessingError("Checkout.com: Failed to process Checkout.com payout") from e

        except HTTPException as e:
            raise e
        except Exception as e:
            log_dict = {
                "message": "Checkout.com: Unexpected error in Checkout.com payout processing",
                "user_id": user_id,
                "amount": str(amount),
                "destination_identifier": destination_identifier,
                "error": str(e),
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentProcessingError("Checkout.com: Payout processing failed") from e

    @staticmethod
    def _map_checkout_status_to_internal(status: str) -> PaymentTransactionStatusEnum:
        """Map Checkout.com's transaction status to our internal PaymentTransactionStatusEnum.

        Args:
            status: The Checkout.com transaction status

        Returns:
            PaymentTransactionStatusEnum: The corresponding internal status
        """
        if status == TransactionStatus.PAID.value:
            return PaymentTransactionStatusEnum.SUCCESS
        elif status == TransactionStatus.DECLINED.value:
            return PaymentTransactionStatusEnum.FAILED
        elif status == TransactionStatus.PENDING.value:
            return PaymentTransactionStatusEnum.PENDING
        else:
            return PaymentTransactionStatusEnum.PENDING

    async def get_payment_status(self, payment_transaction_id: uuid.UUID) -> PaymentResponse:
        log_dict = {
            "message": "Getting Checkout.com payout status",
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

        partner_reference_id = str(transaction.partner_reference_id)
        status = await get_transaction_status(partner_reference_id)
        log_dict = {
            "message": "Checkout.com: Retrieved transaction status",
            "payment_transaction_id": str(payment_transaction_id),
            "status": status,
        }
        logging.info(json_dumps(log_dict))
        return PaymentResponse(
            payment_transaction_id=payment_transaction_id,
            transaction_status=self._map_checkout_status_to_internal(status),
            customer_reference_id=str(payment_transaction_id),
        )

    async def _monitor_transaction_completion(
        self,
        transaction_token: str,
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
        """Monitor Checkout.com payout completion and handle success/failure.

        Args:
            transaction_token: The Checkout.com transaction token to monitor
            payment_transaction_id: The ID of the payment transaction
            points_transaction_id: The ID of the points transaction
            user_id: The user ID
            credits_to_cashout: The number of credits being cashed out
            amount: The amount being transferred
            source_instrument_id: The source payment instrument ID
            destination_instrument_id: The destination payment instrument ID
            destination_identifier: The destination token
            destination_identifier_type: The type of destination identifier
        """
        try:
            start_time = time.time()
            max_wait_time, poll_interval = self.get_polling_config()

            while (time.time() - start_time) < max_wait_time:
                status = await get_transaction_status(transaction_token)
                if status == TransactionStatus.PAID.value:
                    await update_payment_transaction(
                        payment_transaction_id,
                        partner_reference_id=transaction_token,
                        status=PaymentTransactionStatusEnum.SUCCESS,
                        customer_reference_id=transaction_token,
                    )
                    log_dict = {
                        "message": "Checkout.com payout completed",
                        "user_id": user_id,
                        "transaction_token": transaction_token,
                        "status": status,
                        "elapsed_time": time.time() - start_time,
                    }
                    logging.info(json_dumps(log_dict))
                    return
                elif status == TransactionStatus.DECLINED.value:
                    log_dict = {
                        "message": "Checkout.com payout failed",
                        "user_id": user_id,
                        "transaction_token": transaction_token,
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
                "message": ":x: Failure - Checkout.com payout monitoring timed out\n"
                f"transaction_token: {transaction_token}\n"
                f"payment_transaction_id: {payment_transaction_id}\n"
                f"points_transaction_id: {points_transaction_id}\n"
                f"user_id: {user_id}\n"
                f"credits_to_cashout: {credits_to_cashout}\n"
                f"amount: {amount}\n"
                f"source_instrument_id: {source_instrument_id}\n"
                f"destination_instrument_id: {destination_instrument_id}\n"
                f"destination_identifier: {destination_identifier}\n"
                f"destination_identifier_type: {destination_identifier_type}\n"
                f"status: {status}\n",
            }
            logging.error(json_dumps(log_dict))

            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

        except Exception as e:
            log_dict = {
                "message": ":x: Failure - Error monitoring Checkout.com payout completion",
                "user_id": user_id,
                "transaction_token": transaction_token,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
            }
            logging.error(json_dumps(log_dict))
            raise PaymentProcessingError("Failed to monitor Checkout.com payout completion") from e

    @staticmethod
    def get_polling_config() -> tuple[int, int]:
        """
        Returns polling configuration for Checkout.com transactions.
        Returns:
            tuple: (max_wait_time_seconds, poll_interval_seconds)
        """
        return (
            5 * 60,  # 5 minutes in seconds
            10,  # 10 seconds
        )
