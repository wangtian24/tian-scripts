import asyncio
import logging
import os
import time
import uuid
from decimal import Decimal

from plaid.model.transfer_authorization_decision import TransferAuthorizationDecision
from sqlmodel import select
from tenacity import retry, stop_after_attempt, wait_exponential
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.payment.base_types import PaymentInstrumentNotFoundError
from ypl.backend.payment.facilitator import (
    BaseFacilitator,
    PaymentInstrumentError,
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
from ypl.backend.payment.plaid.plaid_payout import (
    PlaidPayout,
    fund_sandbox_account,
    get_balance,
    get_plaid_transfer_status,
    process_plaid_payout,
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

ENVIRONMENT = os.environ.get("ENVIRONMENT")
SYSTEM_USER_ID = "SYSTEM"
SLACK_WEBHOOK_PLAID_CASHOUT = os.getenv("SLACK_WEBHOOK_CASHOUT")
RETRY_ATTEMPTS = 3
RETRY_WAIT_MULTIPLIER = 1
RETRY_WAIT_MIN = 4
RETRY_WAIT_MAX = 15


class PlaidFacilitator(BaseFacilitator):
    def __init__(
        self,
        currency: CurrencyEnum,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        facilitator: PaymentInstrumentFacilitatorEnum,
        destination_additional_details: dict | None = None,
    ):
        super().__init__(currency, destination_identifier_type, facilitator)
        self.destination_additional_details = destination_additional_details

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    )
    async def get_source_instrument_id(self) -> uuid.UUID:
        async with get_async_session() as session:
            query = select(PaymentInstrument).where(
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.PLAID,
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.BANK_ACCOUNT_NUMBER,
                PaymentInstrument.user_id == SYSTEM_USER_ID,
                PaymentInstrument.deleted_at.is_(None),  # type: ignore
            )

            result = await session.exec(query)
            instrument = result.first()
            if not instrument:
                log_dict = {
                    "message": "Source payment instrument not found",
                    "identifier_type": PaymentInstrumentIdentifierTypeEnum.BANK_ACCOUNT_NUMBER,
                    "facilitator": PaymentInstrumentFacilitatorEnum.PLAID,
                    "user_id": SYSTEM_USER_ID,
                }
                logging.exception(json_dumps(log_dict))
                raise PaymentInstrumentNotFoundError(
                    f"Payment instrument not found for {PaymentInstrumentIdentifierTypeEnum.BANK_ACCOUNT_NUMBER}"
                )
            return instrument.payment_instrument_id

    async def get_destination_instrument_id(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        instrument_metadata: dict | None = None,
    ) -> uuid.UUID:
        async with get_async_session() as session:
            query = select(PaymentInstrument).where(
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.PLAID,
                PaymentInstrument.identifier_type == destination_identifier_type,
                PaymentInstrument.identifier == destination_identifier,
                PaymentInstrument.user_id == user_id,
                PaymentInstrument.deleted_at.is_(None),  # type: ignore
            )
            result = await session.exec(query)
            instrument = result.first()
            if not instrument:
                log_dict = {
                    "message": "Destination payment instrument not found. Creating a new one.",
                    "identifier_type": destination_identifier_type,
                    "facilitator": PaymentInstrumentFacilitatorEnum.PLAID,
                    "user_id": user_id,
                    "identifier": destination_identifier,
                }
                logging.info(json_dumps(log_dict))
                instrument = PaymentInstrument(
                    facilitator=PaymentInstrumentFacilitatorEnum.PLAID,
                    identifier_type=destination_identifier_type,
                    identifier=destination_identifier,
                    user_id=user_id,
                )
                session.add(instrument)
                await session.commit()
                return instrument.payment_instrument_id
            return instrument.payment_instrument_id

    async def get_balance(self, currency: CurrencyEnum, payment_transaction_id: uuid.UUID | None = None) -> Decimal:
        plaid_balance = await get_balance()
        return plaid_balance

    async def _send_payment_request(
        self,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        destination_additional_details: dict | None = None,
        *,
        # TODO: Make this a standard parameter.
        payment_transaction_id: uuid.UUID | None = None,
    ) -> PaymentResponse:
        # Flow of the function
        # 0. Get the balance of the source instrument
        # 1. Get the source and destination instrument ids
        # 2. Create a payment transaction
        # 3. Create a point transaction entry for the cashout
        # 4. Update the user points
        # 5. Process the Plaid transfer
        # 6. Monitor the transfer completion
        # 7. If success then update the payment transaction status
        # 8. If failure then reverse the payment transaction, point transaction and update the user points
        start_time = time.time()
        try:
            try:
                # 0. Get the balance of the source instrument
                source_instrument_balance = await self.get_balance(self.currency, payment_transaction_id)

                if source_instrument_balance < amount:
                    if ENVIRONMENT != "production":
                        await fund_sandbox_account()
                    else:
                        log_dict = {
                            "message": "Source instrument does not have enough balance",
                            "user_id": user_id,
                            "amount": str(amount),
                            "source_instrument_balance": str(source_instrument_balance),
                        }
                        logging.error(json_dumps(log_dict))
                        raise ValueError("Source instrument does not have enough balance")

                source_instrument_id = await self.get_source_instrument_id()

                plaid_destination_identifier = str(
                    (destination_additional_details or {}).get("account_number", "")
                    + "-"
                    + (destination_additional_details or {}).get("routing_number", "")
                    + "-"
                    + (destination_additional_details or {}).get("account_type", "")
                )
                destination_instrument_id = await self.get_destination_instrument_id(
                    user_id, plaid_destination_identifier, destination_identifier_type
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
                    "message": "Failed to update user points or transaction status",
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
                raise PaymentProcessingError("Failed to update user points or transaction status") from e

            try:
                plaid_payout = PlaidPayout(
                    user_id=user_id,
                    user_name=(destination_additional_details or {}).get("user_name", ""),
                    amount=amount,
                    account_number=(destination_additional_details or {}).get("account_number", ""),
                    routing_number=(destination_additional_details or {}).get("routing_number", ""),
                    account_type=(destination_additional_details or {}).get("account_type", ""),
                )
                transfer_id, transfer_status = await process_plaid_payout(plaid_payout)

                await update_payment_transaction(
                    payment_transaction_id,
                    partner_reference_id=transfer_id,
                    status=PaymentTransactionStatusEnum.SUCCESS,
                    customer_reference_id=transfer_id,
                )

                # Start monitoring in background task
                asyncio.create_task(
                    self._monitor_transfer_completion(
                        transfer_id=transfer_id,
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
                    "message": "Successfully submitted for Plaid transfer",
                    "duration": str(end_time - start_time),
                    "user_id": user_id,
                    "amount": str(amount),
                    "credits_to_cashout": str(credits_to_cashout),
                    "source_instrument_id": str(source_instrument_id),
                    "destination_instrument_id": str(destination_instrument_id),
                    "destination_identifier": destination_identifier,
                    "currency": self.currency.value,
                    "transfer_id": transfer_id,
                }
                logging.info(json_dumps(log_dict))
                asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_PLAID_CASHOUT))
                return PaymentResponse(
                    payment_transaction_id=payment_transaction_id,
                    transaction_status=PaymentTransactionStatusEnum.PENDING,
                    customer_reference_id=transfer_id,
                )

            except Exception as e:
                log_dict = {
                    "message": "Failed to process Plaid transfer. Reversing transaction.",
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
                raise PaymentProcessingError("Failed to process Plaid transfer") from e

        except Exception as e:
            log_dict = {
                "message": "Unexpected error in payment processing",
                "user_id": user_id,
                "amount": str(amount),
                "destination_identifier": destination_identifier,
                "error": str(e),
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentProcessingError("Payment processing failed") from e

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
                "message": "Failed to process Plaid transfer. Reversing transaction.",
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
            asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_PLAID_CASHOUT))

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
                    "destination_identifier_type": str(destination_identifier_type),
                    "reversal_transaction_id": str(payment_transaction_id),
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
                "message": "Successfully reversed transaction",
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
            asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_PLAID_CASHOUT))
        except Exception as e:
            error_message = str(e)
            log_dict = {
                "message": "Failed to handle failed transaction cleanup",
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
            asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_PLAID_CASHOUT))

    async def get_payment_status(self, payment_transaction_id: uuid.UUID) -> PaymentResponse:
        # TODO: Implement this
        return PaymentResponse(
            payment_transaction_id=payment_transaction_id,
            transaction_status=PaymentTransactionStatusEnum.SUCCESS,
            customer_reference_id=str(payment_transaction_id),
        )

    async def _monitor_transfer_completion(
        self,
        transfer_id: str,
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
        """Monitor transfer completion and handle success/failure.

        Args:
            transfer_id: The Plaid transfer ID to monitor
            payment_transaction_id: The ID of the payment transaction
            points_transaction_id: The ID of the points transaction
            user_id: The user ID
            credits_to_cashout: The number of credits being cashed out
            amount: The amount being transferred
            source_instrument_id: The source payment instrument ID
            destination_instrument_id: The destination payment instrument ID
            destination_identifier: The destination account ID
            destination_identifier_type: The type of destination identifier
        """
        try:
            start_time = time.time()
            # TODO: Refactor this so that we don't use long polling
            # and instead rely on a webhook or cron job to check for transfer status
            max_wait_time, poll_interval = self.get_polling_config()

            while (time.time() - start_time) < max_wait_time:
                status = await get_plaid_transfer_status(transfer_id)

                if status == TransferAuthorizationDecision.allowed_values[("value",)]["APPROVED"]:
                    await update_payment_transaction(
                        payment_transaction_id,
                        partner_reference_id=transfer_id,
                        status=PaymentTransactionStatusEnum.SUCCESS,
                        customer_reference_id=transfer_id,
                    )
                    log_dict = {
                        "message": "Plaid transfer completed",
                        "user_id": user_id,
                        "transfer_id": transfer_id,
                        "status": status,
                        "elapsed_time": time.time() - start_time,
                    }
                    logging.info(json_dumps(log_dict))
                    return
                elif status == TransferAuthorizationDecision.allowed_values[("value",)]["DECLINED"]:
                    log_dict = {
                        "message": "Plaid transfer declined",
                        "user_id": user_id,
                        "transfer_id": transfer_id,
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
                "message": "Plaid transfer monitoring timed out",
                "user_id": user_id,
                "transfer_id": transfer_id,
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
                "message": "Error monitoring transfer completion",
                "user_id": user_id,
                "transfer_id": transfer_id,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
            }
            logging.error(json_dumps(log_dict))
            raise PaymentProcessingError("Failed to monitor transfer completion") from e

    @staticmethod
    def get_polling_config() -> tuple[int, int]:
        """
        Returns polling configuration for Plaid transfers.
        Returns:
            tuple: (max_wait_time_seconds, poll_interval_seconds)
        """
        return (
            72 * 60 * 60,  # 72 hours in seconds
            24 * 60 * 60,  # 24 hours in seconds
        )
