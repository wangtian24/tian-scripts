import asyncio
import logging
import os
import time
import uuid
from decimal import Decimal

from sqlmodel import select
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack_with_user_name
from ypl.backend.payment.facilitator import (
    BaseFacilitator,
    PaymentInstrumentError,
    PaymentProcessingError,
    PaymentResponse,
    PaymentStatusFetchError,
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
    handle_failed_transaction,
)
from ypl.backend.payment.payout_utils import (
    get_source_instrument_id as get_generic_source_instrument_id,
)
from ypl.backend.payment.paypal.paypal_payout import (
    PayPalPayout,
    PayPalPayoutError,
    TransactionStatus,
    get_transaction_status,
    process_paypal_payout,
)
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
SLACK_WEBHOOK_CASHOUT = os.getenv("SLACK_WEBHOOK_CASHOUT", "")
RETRY_ATTEMPTS = 3
RETRY_WAIT_MULTIPLIER = 1
RETRY_WAIT_MIN = 4
RETRY_WAIT_MAX = 15


class PayPalFacilitator(BaseFacilitator):
    def __init__(
        self,
        currency: CurrencyEnum,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        facilitator: PaymentInstrumentFacilitatorEnum,
    ):
        super().__init__(currency, destination_identifier_type, facilitator)

    async def get_source_instrument_id(self) -> uuid.UUID:
        return await get_generic_source_instrument_id(
            PaymentInstrumentFacilitatorEnum.PAYPAL,
            PaymentInstrumentIdentifierTypeEnum.PARTNER_IDENTIFIER,
        )

    async def get_destination_instrument_id(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        instrument_metadata: dict | None = None,
    ) -> uuid.UUID:
        return await get_destination_instrument_id(
            PaymentInstrumentFacilitatorEnum.PAYPAL,
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
        # TODO: Implement this
        return Decimal(0)

    async def _send_payment_request(
        self,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
        usd_amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        destination_additional_details: dict | None = None,
        payment_transaction_id: uuid.UUID | None = None,
    ) -> PaymentResponse:
        start_time = time.time()

        try:
            try:
                credits_to_cashout += CASHOUT_TXN_COST
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
                    "message": "PayPal: Failed to create payment transaction",
                    "user_id": user_id,
                    "amount": str(amount),
                    "usd_amount": str(usd_amount),
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                raise PaymentProcessingError("PayPal: Failed to create payment transaction") from e

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
                    "message": "PayPal: Failed to create point transaction",
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
                raise PaymentProcessingError("PayPal: Failed to create point transaction") from e

            try:
                await update_user_points(user_id, -credits_to_cashout)
            except Exception as e:
                log_dict = {
                    "message": "PayPal: Failed to update user points",
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
                raise PaymentProcessingError("PayPal: Failed to update user points") from e

            try:
                paypal_payout = PayPalPayout(
                    amount=amount,
                    payment_transaction_id=payment_transaction_id,
                    currency=self.currency,
                    destination_type=destination_identifier_type,
                    destination_identifier=destination_identifier,
                )
                batch_id, batch_status = await process_paypal_payout(paypal_payout)
                transaction_status = await get_transaction_status(batch_id)

                await update_payment_transaction(
                    payment_transaction_id,
                    partner_reference_id=batch_id,
                    status=self.map_paypal_status_to_internal(transaction_status),
                    customer_reference_id=batch_id,
                )

                if transaction_status not in (TransactionStatus.SUCCESS):
                    # Start monitoring in background task
                    asyncio.create_task(
                        self._monitor_transaction_completion(
                            batch_id=batch_id,
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
                    "message": ":white_check_mark: Success - PayPal payout submitted",
                    "duration": str(end_time - start_time),
                    "user_id": user_id,
                    "amount": str(amount),
                    "credits_to_cashout": str(credits_to_cashout),
                    "source_instrument_id": str(source_instrument_id),
                    "destination_instrument_id": str(destination_instrument_id),
                    "destination_identifier": destination_identifier,
                    "currency": self.currency.value,
                    "batch_id": batch_id,
                    "payment_transaction_id": str(payment_transaction_id),
                    "points_transaction_id": str(point_transaction_id),
                }
                logging.info(json_dumps(log_dict))
                asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
                return PaymentResponse(
                    payment_transaction_id=payment_transaction_id,
                    transaction_status=PaymentTransactionStatusEnum.PENDING,
                    customer_reference_id=batch_id,
                )

            except PayPalPayoutError as e:
                log_dict = {
                    "message": "PayPal: Failed to process PayPal payout",
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
                    "message": "PayPal: Failed to process PayPal payout. Reversing transaction.",
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
                raise PaymentProcessingError("PayPal: Failed to process PayPal payout") from e

        except Exception as e:
            log_dict = {
                "message": "PayPal: Unexpected error in PayPal payout processing",
                "user_id": user_id,
                "amount": str(amount),
                "destination_identifier": destination_identifier,
                "error": str(e),
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentProcessingError("PayPal: Payout processing failed") from e

    @staticmethod
    def map_paypal_status_to_internal(status: str) -> PaymentTransactionStatusEnum:
        """Map PayPal's transaction status to our internal PaymentTransactionStatusEnum.

        Args:
            status: The PayPal transaction status

        Returns:
            PaymentTransactionStatusEnum: The corresponding internal status
        """
        if status == TransactionStatus.SUCCESS:
            return PaymentTransactionStatusEnum.SUCCESS
        elif status in (
            TransactionStatus.FAILED,
            TransactionStatus.RETURNED,
            TransactionStatus.REFUNDED,
            TransactionStatus.REVERSED,
        ):
            return PaymentTransactionStatusEnum.FAILED
        else:
            return PaymentTransactionStatusEnum.PENDING

    async def get_payment_status(self, payment_transaction_id: uuid.UUID) -> PaymentResponse:
        log_dict = {
            "message": "Getting PayPal payout status",
            "payment_transaction_id": str(payment_transaction_id),
            "currency": self.currency.value,
        }
        async with get_async_session() as session:
            query = select(PaymentTransaction).where(
                PaymentTransaction.payment_transaction_id == payment_transaction_id
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
            "message": "PayPal: Retrieved transaction status",
            "payment_transaction_id": str(payment_transaction_id),
            "status": status,
        }
        logging.info(json_dumps(log_dict))
        return PaymentResponse(
            payment_transaction_id=payment_transaction_id,
            transaction_status=self.map_paypal_status_to_internal(status),
            customer_reference_id=str(transaction.customer_reference_id),
        )

    async def _monitor_transaction_completion(
        self,
        batch_id: str,
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
        """Monitor PayPal payout completion and handle success/failure.

        Args:
            batch_id: The PayPal batch ID to monitor
            payment_transaction_id: The ID of the payment transaction
            points_transaction_id: The ID of the points transaction
            user_id: The user ID
            credits_to_cashout: The number of credits being cashed out
            amount: The amount being transferred
            source_instrument_id: The source payment instrument ID
            destination_instrument_id: The destination payment instrument ID
            destination_identifier: The destination identifier
            destination_identifier_type: The type of destination identifier
        """
        try:
            start_time = time.time()
            max_wait_time, poll_interval = self.get_polling_config()

            while (time.time() - start_time) < max_wait_time:
                status = await get_transaction_status(batch_id)
                if status == TransactionStatus.SUCCESS:
                    await update_payment_transaction(
                        payment_transaction_id,
                        partner_reference_id=batch_id,
                        status=PaymentTransactionStatusEnum.SUCCESS,
                        customer_reference_id=batch_id,
                    )
                    log_dict = {
                        "message": "PayPal payout completed",
                        "user_id": user_id,
                        "batch_id": batch_id,
                        "status": status,
                        "elapsed_time": time.time() - start_time,
                    }
                    logging.info(json_dumps(log_dict))
                    return
                elif status in (
                    TransactionStatus.FAILED,
                    TransactionStatus.RETURNED,
                    TransactionStatus.REFUNDED,
                    TransactionStatus.REVERSED,
                ):
                    log_dict = {
                        "message": "PayPal payout failed",
                        "user_id": user_id,
                        "batch_id": batch_id,
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
                "message": ":x: Failure - PayPal payout monitoring timed out\n"
                f"batch_id: {batch_id}\n"
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

            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

        except Exception as e:
            log_dict = {
                "message": ":x: Failure - Error monitoring PayPal payout completion",
                "user_id": user_id,
                "batch_id": batch_id,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
            }
            logging.error(json_dumps(log_dict))
            raise PaymentProcessingError("Failed to monitor PayPal payout completion") from e

    @staticmethod
    def get_polling_config() -> tuple[int, int]:
        """
        Returns polling configuration for PayPal transactions.
        Returns:
            tuple: (max_wait_time_seconds, poll_interval_seconds)
        """
        return (
            1 * 60,  # 1 minutes in seconds
            10,  # 10 seconds
        )
