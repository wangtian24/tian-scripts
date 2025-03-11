import asyncio
import json
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException
from sqlalchemy.orm import selectinload
from sqlmodel import select, update
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack, post_to_slack_with_user_name
from ypl.backend.payment.base_types import (
    PaymentInstrumentError,
    PaymentProcessingError,
    PaymentResponse,
    PaymentStatusFetchError,
    UpiDestinationMetadata,
    ValidateDestinationIdentifierResponse,
)
from ypl.backend.payment.facilitator import BaseFacilitator
from ypl.backend.payment.payout_utils import (
    SLACK_WEBHOOK_CASHOUT,
    get_destination_instrument_id,
    get_source_instrument_id,
)
from ypl.backend.payment.upi.axis.request_utils import (
    AxisPaymentRequest,
    get_balance,
    get_payment_status,
    make_payment,
    verify_vpa,
)
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import (
    CurrencyEnum,
    PaymentInstrumentFacilitatorEnum,
    PaymentInstrumentIdentifierTypeEnum,
    PaymentTransaction,
    PaymentTransactionStatusEnum,
)
from ypl.db.point_transactions import PointsActionEnum, PointTransaction
from ypl.db.users import User

VALIDATE_DESTINATION_IDENTIFIER_TOKEN_TTL_SECONDS = 5 * 60
MIN_BALANCE_FOR_ALERT = Decimal("20_000")  # 20,000 INR


def _mask_name(name: str) -> str:
    words = name.split()
    masked_words = []
    for word in words:
        if len(word) <= 2:
            masked_words.append(word)
        else:
            masked_words.append(f"{word[:2]}{'*' * (len(word)-3)}{word[-1]}")
    return " ".join(masked_words)


def log_payment_retry_attempt(retry_state: RetryCallState) -> None:
    """Custom logger for payment retry attempts"""
    payment_transaction_id = retry_state.kwargs.get("payment_transaction_id")
    partner_reference_id = retry_state.kwargs.get("partner_reference_id")
    user_id = retry_state.kwargs.get("user_id")
    start_time = retry_state.kwargs.get("start_time")

    logging.info(
        json_dumps(
            {
                "message": f"Retrying payment check in attempt {retry_state.attempt_number}",
                "attempt_number": retry_state.attempt_number,
                "sleep_time": retry_state.next_action.sleep if retry_state.next_action else None,
                "payment_transaction_id": str(payment_transaction_id),
                "partner_reference_id": partner_reference_id,
                "user_id": user_id,
                "error": str(retry_state.outcome.exception()) if retry_state.outcome else None,
                "start_time": start_time,
                "elapsed_time": time.time() - start_time,
                "facilitator": "UPI",
            }
        )
    )


def get_retry_decorator(max_attempts: int = 35, multiplier: float = 1, min_wait: float = 1, max_wait: float = 10):  # type: ignore[no-untyped-def]
    return retry(
        retry=retry_if_exception_type(PaymentStatusFetchError),
        wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
        stop=stop_after_attempt(max_attempts),
        before_sleep=log_payment_retry_attempt,
    )


class AxisUpiFacilitator(BaseFacilitator):
    # Only used as a placeholder. This ID will not be sent to the bank during the payment request.
    SOURCE_INSTRUMENT_UPI_ID = "AXIS"

    async def get_balance(self, currency: CurrencyEnum, payment_transaction_id: uuid.UUID | None = None) -> Decimal:
        balance = await get_balance(payment_transaction_id)
        if balance < MIN_BALANCE_FOR_ALERT and settings.ENVIRONMENT == "production":
            message = (
                f":red_circle: *Axis account balance low alert for UPI*\n"
                f"Current balance: INR {balance}\n"
                f"Minimum required: INR {MIN_BALANCE_FOR_ALERT}"
            )
            asyncio.create_task(post_to_slack(message))
        return balance

    async def get_source_instrument_id(self) -> uuid.UUID:
        return await get_source_instrument_id(
            PaymentInstrumentFacilitatorEnum.UPI,
            PaymentInstrumentIdentifierTypeEnum.UPI_ID,
            self.SOURCE_INSTRUMENT_UPI_ID,
        )

    async def get_destination_instrument_id(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        instrument_metadata: dict | None = None,
    ) -> uuid.UUID:
        return await get_destination_instrument_id(
            PaymentInstrumentFacilitatorEnum.UPI,
            user_id,
            destination_identifier,
            destination_identifier_type,
            instrument_metadata,
        )

    # TODO: Move this to the base facilitator.
    async def undo_payment_transaction(
        self, payment_transaction_id: uuid.UUID, customer_reference_id: str | None = None
    ) -> None:
        """
        Undo a payment transaction.

        Args:
            payment_transaction_id: The ID of the payment transaction to undo.
            customer_reference_id: The customer reference ID returned by the partner.
                If provided, it will be stored along with the failed payment transaction.
        """
        try:
            async with get_async_session() as session:
                async with session.begin():
                    # Set the isolation level to SERIALIZABLE to prevent concurrent
                    # updates of all the payment transactions.
                    await session.connection(execution_options={"isolation_level": "SERIALIZABLE"})

                    # 1. Find the payment transaction and its associated point/credit transaction.
                    payment_transaction = (
                        await session.exec(
                            select(PaymentTransaction)
                            .options(selectinload(PaymentTransaction.destination_instrument))  # type: ignore
                            .options(selectinload(PaymentTransaction.credits_transaction))  # type: ignore
                            .where(
                                PaymentTransaction.payment_transaction_id == payment_transaction_id,
                                PaymentTransaction.deleted_at.is_(None),  # type: ignore
                            )
                        )
                    ).one()

                    # Ensure that it is not in SUCCESS or FAILED state.
                    if payment_transaction.status in [
                        PaymentTransactionStatusEnum.SUCCESS,
                        PaymentTransactionStatusEnum.FAILED,
                    ]:
                        log_dict = {
                            "message": (
                                f"Could not undo payment transaction. "
                                f"Payment transaction already in the final {payment_transaction.status} state"
                            ),
                            "payment_transaction_id": str(payment_transaction_id),
                            "customer_reference_id": customer_reference_id,
                            "facilitator": self.facilitator,
                            "user_id": str(payment_transaction.destination_instrument.user_id),
                        }
                        logging.info(json_dumps(log_dict))
                        raise ValueError("Payment transaction already in the final state")

                    point_transaction = payment_transaction.credits_transaction

                    log_dict = {
                        "message": ":x: Failure - Failed to process payout reward. Reversing transaction.",
                        "user_id": str(point_transaction.user_id),
                        "payment_transaction_id": str(payment_transaction_id),
                        "customer_reference_id": customer_reference_id,
                        "points_transaction_id": str(point_transaction.transaction_id),
                        "credits_to_cashout": str(point_transaction.point_delta),
                        "amount": str(payment_transaction.amount),
                        "usd_amount": str(payment_transaction.usd_amount),
                        "source_instrument_id": str(payment_transaction.source_instrument_id),
                        "destination_instrument_id": str(payment_transaction.destination_instrument_id),
                        "destination_identifier": payment_transaction.destination_instrument.identifier,
                        "destination_identifier_type": str(payment_transaction.destination_instrument.identifier_type),
                        "facilitator": self.facilitator,
                    }
                    logging.info(json_dumps(log_dict))
                    asyncio.create_task(
                        post_to_slack_with_user_name(
                            str(point_transaction.user_id), json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT
                        )
                    )

                    # 2. Update the user's points.
                    # point_delta is negative, so we add it to the user's points.
                    await session.exec(
                        update(User)
                        .values(points=User.points - point_transaction.point_delta)
                        .where(User.user_id == point_transaction.user_id)  # type: ignore
                    )

                    # 3. Add the point transaction reversal record.
                    # point_delta is negative, so we invert it to make it positive.
                    session.add(
                        PointTransaction(
                            user_id=point_transaction.user_id,
                            point_delta=-point_transaction.point_delta,
                            action_type=PointsActionEnum.CASHOUT_REVERSED,
                            cashout_payment_transaction_id=payment_transaction_id,
                        )
                    )

                    # 4. Update the payment transaction status to FAILED.
                    payment_transaction.status = PaymentTransactionStatusEnum.FAILED
                    payment_transaction.last_status_change_at = datetime.now()
                    if customer_reference_id is not None:
                        payment_transaction.customer_reference_id = customer_reference_id

                    session.add(payment_transaction)

                    # 5. Create a reversal payment transaction.
                    session.add(
                        PaymentTransaction(
                            payment_transaction_id=uuid.uuid4(),
                            currency=payment_transaction.currency,
                            amount=payment_transaction.amount,
                            usd_amount=payment_transaction.usd_amount,
                            source_instrument_id=payment_transaction.source_instrument_id,
                            destination_instrument_id=payment_transaction.destination_instrument_id,
                            status=PaymentTransactionStatusEnum.REVERSED,
                            last_status_change_at=datetime.now(),
                        )
                    )
            log_dict = {
                "message": ":white_check_mark: Success - Reversed transaction",
                "payment_transaction_id": str(payment_transaction_id),
                "customer_reference_id": customer_reference_id,
                "points_transaction_id": str(point_transaction.transaction_id),
                "user_id": str(point_transaction.user_id),
                "amount": str(payment_transaction.amount),
                "usd_amount": str(payment_transaction.usd_amount),
                "source_instrument_id": str(payment_transaction.source_instrument_id),
                "destination_instrument_id": str(payment_transaction.destination_instrument_id),
                "destination_identifier": payment_transaction.destination_instrument.identifier,
                "destination_identifier_type": str(payment_transaction.destination_instrument.identifier_type),
                "facilitator": self.facilitator,
            }
            logging.info(json_dumps(log_dict))
            asyncio.create_task(
                post_to_slack_with_user_name(
                    str(point_transaction.user_id), json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT
                )
            )
        except Exception as e:
            log_dict = {
                "message": ":x: Failure - Could not undo payment transaction",
                "payment_transaction_id": str(payment_transaction_id),
                "customer_reference_id": customer_reference_id,
                "error": str(e),
                "facilitator": self.facilitator,
            }
            logging.exception(json_dumps(log_dict))
            asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

    async def _check_payment_status_with_retry(
        self,
        payment_transaction_id: uuid.UUID,
        partner_reference_id: str,
        user_id: str,
        current_status: PaymentTransactionStatusEnum,
        start_time: float,
        retry_config: dict,
    ) -> None:
        retry_decorator = get_retry_decorator(**retry_config)
        await retry_decorator(self._check_payment_status)(
            payment_transaction_id=payment_transaction_id,
            partner_reference_id=partner_reference_id,
            user_id=user_id,
            current_status=current_status,
            start_time=start_time,
        )

    # Current status can be NOT_STARTED and PENDING.
    # It will be in NOT_STARTED if we do not know the status of the make_payment call.
    # In the case of NOT_STARTED, we update the status to PENDING if the payment is pending.
    # In the case of PENDING, we check if the payment is still pending.
    # This call is made in a loop until the payment is in the final state, so the current_status can be the same even if
    # there is a change in the db state (e.g., the payment is moved to PENDING state from NOT_STARTED).
    # TODO: Use a workflow / state machine to track the payment status instead of blind retries.
    async def _check_payment_status(
        self,
        *,  # Force keyword arguments
        payment_transaction_id: uuid.UUID,
        partner_reference_id: str,
        user_id: str,
        current_status: PaymentTransactionStatusEnum,
        start_time: float,
    ) -> None:
        attempt_start_time = time.time()
        try:
            status = await get_payment_status(payment_transaction_id, partner_reference_id)
        except Exception as e:
            log_dict = {
                "message": "Failed to get payment status. Retrying after a delay.",
                "payment_transaction_id": str(payment_transaction_id),
                "partner_reference_id": partner_reference_id,
                "user_id": user_id,
                "current_status": current_status,
                "error": str(e),
                "facilitator": self.facilitator,
                "elapsed_time": time.time() - start_time,
                "attempt_start_time": attempt_start_time,
                "attempt_time": time.time() - attempt_start_time,
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentStatusFetchError("Failed to get payment status") from e

        if (
            current_status == PaymentTransactionStatusEnum.NOT_STARTED
            and status.transaction_status == PaymentTransactionStatusEnum.PENDING
        ):
            # TODO: Avoid updating this multiple times during the retry loop.
            # Read the note above this method for more details.
            async with get_async_session() as session:
                async with session.begin():
                    await session.exec(
                        update(PaymentTransaction)
                        .values(
                            status=PaymentTransactionStatusEnum.PENDING,
                            last_status_change_at=datetime.now(),
                        )
                        .where(
                            PaymentTransaction.payment_transaction_id == payment_transaction_id,  # type: ignore
                            # This check is needed to avoid updating the status to pending
                            # if it is already in the final state.
                            PaymentTransaction.status == PaymentTransactionStatusEnum.NOT_STARTED,  # type: ignore
                        )
                    )
            log_dict = {
                "message": "Payment moved to pending state",
                "payment_transaction_id": str(payment_transaction_id),
                "partner_reference_id": partner_reference_id,
                "user_id": user_id,
                "current_status": current_status,
                "elapsed_time": time.time() - start_time,
                "attempt_start_time": attempt_start_time,
                "attempt_time": time.time() - attempt_start_time,
                "facilitator": self.facilitator,
            }
            logging.info(json_dumps(log_dict))
            raise PaymentStatusFetchError("Payment moved to pending state")
        elif status.transaction_status == PaymentTransactionStatusEnum.PENDING:
            log_dict = {
                "message": "Payment is still pending",
                "payment_transaction_id": str(payment_transaction_id),
                "partner_reference_id": partner_reference_id,
                "user_id": user_id,
                "current_status": current_status,
                "elapsed_time": time.time() - start_time,
                "attempt_start_time": attempt_start_time,
                "attempt_time": time.time() - attempt_start_time,
                "facilitator": self.facilitator,
            }
            logging.info(json_dumps(log_dict))
            raise PaymentStatusFetchError("Payment is still pending")
        elif status.transaction_status == PaymentTransactionStatusEnum.FAILED:
            # Axis returns a customer reference ID for failed transactions as well, store it with the failed transaction
            await self.undo_payment_transaction(payment_transaction_id, status.customer_reference_id)
            log_dict = {
                "message": ":x: Failure - Payment failed",
                "payment_transaction_id": str(payment_transaction_id),
                "partner_reference_id": partner_reference_id,
                "user_id": user_id,
                "elapsed_time": time.time() - start_time,
                "attempt_start_time": attempt_start_time,
                "attempt_time": time.time() - attempt_start_time,
                "facilitator": self.facilitator,
                "end_time": time.time(),
            }
            logging.error(json_dumps(log_dict))
            asyncio.create_task(self._post_to_slack_with_destination_details(user_id, log_dict, payment_transaction_id))
            raise PaymentProcessingError("Payment failed. Finishing payment monitoring")
        elif status.transaction_status == PaymentTransactionStatusEnum.SUCCESS:
            db_update_start_time = time.time()
            try:
                async with get_async_session() as session:
                    async with session.begin():
                        await session.exec(
                            update(PaymentTransaction)
                            .values(
                                status=PaymentTransactionStatusEnum.SUCCESS,
                                last_status_change_at=datetime.now(),
                                customer_reference_id=status.customer_reference_id,
                            )
                            .where(PaymentTransaction.payment_transaction_id == payment_transaction_id)  # type: ignore
                        )
            except Exception as e:
                log_dict = {
                    "message": ":x: Failure - Failed to update payment transaction status to SUCCESS",
                    "payment_transaction_id": str(payment_transaction_id),
                    "partner_reference_id": partner_reference_id,
                    "user_id": user_id,
                    "error": str(e),
                    "facilitator": self.facilitator,
                    "elapsed_time": time.time() - start_time,
                    "attempt_start_time": attempt_start_time,
                    "attempt_time": time.time() - attempt_start_time,
                    "db_update_start_time": db_update_start_time,
                    "db_update_time": time.time() - db_update_start_time,
                }
                logging.exception(json_dumps(log_dict))
                raise PaymentStatusFetchError("Failed to update payment transaction status to SUCCESS") from e
            log_dict = {
                "message": ":white_check_mark: Success - Payment successful",
                "payment_transaction_id": str(payment_transaction_id),
                "partner_reference_id": partner_reference_id,
                "user_id": user_id,
                "elapsed_time": time.time() - start_time,
                "attempt_start_time": attempt_start_time,
                "attempt_time": time.time() - attempt_start_time,
                "db_update_start_time": db_update_start_time,
                "db_update_time": time.time() - db_update_start_time,
                "facilitator": self.facilitator,
                "end_time": time.time(),
            }
            logging.info(json_dumps(log_dict))
            asyncio.create_task(self._post_to_slack_with_destination_details(user_id, log_dict, payment_transaction_id))
            return

    async def _post_to_slack_with_destination_details(
        self, user_id: str, log_dict: dict, payment_transaction_id: uuid.UUID
    ) -> None:
        async with get_async_session() as session:
            payment_transaction = (
                await session.exec(
                    select(PaymentTransaction)
                    .options(selectinload(PaymentTransaction.destination_instrument))  # type: ignore
                    .where(PaymentTransaction.payment_transaction_id == payment_transaction_id)
                )
            ).one()
        log_dict["destination_identifier"] = payment_transaction.destination_instrument.identifier
        log_dict["destination_identifier_type"] = payment_transaction.destination_instrument.identifier_type
        asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

    # TODO: Make this generic, and resilient
    async def monitor_payment_status(
        self,
        payment_transaction_id: uuid.UUID,
        partner_reference_id: str,
        user_id: str,
        current_status: PaymentTransactionStatusEnum,
        retry_config: dict | None = None,
    ) -> None:
        start_time = time.time()

        # Retry starting with 1s delay, and doubling the delay exponentially up to 10s.
        # Waiting up to ~5m excluding the API call time.
        retry_config = retry_config or {"max_attempts": 35, "multiplier": 1, "min_wait": 1, "max_wait": 10}

        log_dict: dict[str, Any] = {
            "message": "Payment monitoring started",
            "payment_transaction_id": str(payment_transaction_id),
            "partner_reference_id": partner_reference_id,
            "user_id": user_id,
            "current_status": current_status,
            "facilitator": self.facilitator,
            "retry_config": retry_config,
            "start_time": start_time,
        }
        logging.info(json_dumps(log_dict))

        try:
            await self._check_payment_status_with_retry(
                payment_transaction_id=payment_transaction_id,
                partner_reference_id=partner_reference_id,
                user_id=user_id,
                current_status=current_status,
                start_time=start_time,
                retry_config=retry_config,
            )
        except Exception as e:
            # If we get here, we've timed out
            # Do not reverse the transaction here as the txn might still complete
            log_dict = {
                "message": ":x: Failure - Axis UPI payment monitoring timed out",
                "payment_transaction_id": str(payment_transaction_id),
                "user_id": user_id,
                "partner_reference_id": partner_reference_id,
                "error": str(e),
                "facilitator": self.facilitator,
                "elapsed_time": time.time() - start_time,
            }
            logging.exception(json_dumps(log_dict))
            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
            raise PaymentStatusFetchError("Failed to monitor payment status") from e

    def _get_validated_destination_details(self, validated_destination_details: str) -> dict:
        decryptor = Fernet(settings.VALIDATE_DESTINATION_IDENTIFIER_SECRET_KEY)
        decrypted_string = decryptor.decrypt_at_time(
            validated_destination_details.encode("utf-8"),
            VALIDATE_DESTINATION_IDENTIFIER_TOKEN_TTL_SECONDS,
            int(datetime.now(UTC).timestamp()),
        ).decode("utf-8")
        return json.loads(decrypted_string)  # type: ignore[no-any-return]

    # TODO: Move this to the base facilitator.
    async def make_payment(
        self,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
        usd_amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        destination_additional_details: dict | None = None,
        validated_destination_details: str | None = None,
    ) -> PaymentResponse:
        start_time = time.time()
        # TODO: Get this from where the payment is initiated (i.e., from the API handler).
        payment_transaction_id = uuid.uuid4()

        instrument_metadata = destination_additional_details

        if validated_destination_details is not None:
            if instrument_metadata is None:
                instrument_metadata = {}
            try:
                instrument_metadata.update(self._get_validated_destination_details(validated_destination_details))
            except Exception as e:
                log_dict = {
                    "message": "Failed to decrypt validated destination details",
                    "error": str(e),
                    "user_id": user_id,
                    "destination_identifier": destination_identifier,
                    "destination_identifier_type": destination_identifier_type,
                    "credits_to_cashout": credits_to_cashout,
                    "amount": str(amount),
                    "usd_amount": str(usd_amount),
                    "currency": self.currency,
                    "payment_transaction_id": str(payment_transaction_id),
                    "facilitator": self.facilitator,
                    "elapsed_time": time.time() - start_time,
                }
                logging.exception(json_dumps(log_dict))
                if isinstance(e, InvalidToken):
                    raise PaymentInstrumentError("Expired validated destination details, please try again") from e
                raise PaymentInstrumentError("Failed to get validated destination details") from e

        try:
            # 0. Get the balance of the source instrument and verify if it is enough.
            # 1. Get the source and destination instrument IDs.
            balance, source_instrument_id, destination_instrument_id = await asyncio.gather(
                self.get_balance(self.currency),
                self.get_source_instrument_id(),
                self.get_destination_instrument_id(
                    user_id,
                    destination_identifier,
                    destination_identifier_type,
                    instrument_metadata,
                ),
            )
            if balance < amount:
                log_dict = {
                    "message": ":x: Failure - Source instrument does not have enough balance",
                    "user_id": user_id,
                    "credits_to_cashout": credits_to_cashout,
                    "amount": str(amount),
                    "usd_amount": str(usd_amount),
                    "currency": self.currency,
                    "source_instrument_balance": str(balance),
                    "facilitator": self.facilitator,
                    "payment_transaction_id": str(payment_transaction_id),
                    "processing_time": time.time() - start_time,
                    "elapsed_time": time.time() - start_time,
                    "destination_identifier": destination_identifier,
                    "destination_identifier_type": destination_identifier_type,
                }
                logging.error(json_dumps(log_dict))
                asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
                raise ValueError("Source instrument does not have enough balance")
        except (ValueError, HTTPException):
            # Bubble up these exceptions as they are expected to be handled at the API level.
            raise
        except Exception as e:
            log_dict = {
                "message": ":x: Failure - Failed to initiate make_payment",
                "user_id": user_id,
                "error": str(e),
                "facilitator": self.facilitator,
                "payment_transaction_id": str(payment_transaction_id),
                "credits_to_cashout": credits_to_cashout,
                "amount": str(amount),
                "usd_amount": str(usd_amount),
                "currency": self.currency,
                "processing_time": time.time() - start_time,
                "elapsed_time": time.time() - start_time,
                "destination_identifier": destination_identifier,
                "destination_identifier_type": destination_identifier_type,
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentInstrumentError("Failed to get payment instruments") from e
        balance_validation_done = time.time()
        log_dict = {
            "message": "Successfully got payment instruments and validated balance",
            "user_id": user_id,
            "payment_transaction_id": str(payment_transaction_id),
            "processing_time": balance_validation_done - start_time,
            "elapsed_time": time.time() - start_time,
            "credits_to_cashout": credits_to_cashout,
            "amount": str(amount),
            "usd_amount": str(usd_amount),
            "currency": self.currency,
            "facilitator": self.facilitator,
            "destination_identifier": destination_identifier,
            "destination_identifier_type": destination_identifier_type,
        }
        logging.info(json_dumps(log_dict))

        try:
            async with get_async_session() as session:
                async with session.begin():
                    # Set the isolation level to SERIALIZABLE to prevent concurrent
                    # updates of all the payment transactions.
                    await session.connection(execution_options={"isolation_level": "SERIALIZABLE"})

                    # Ensure that the user has enough points to cash out before starting the payment steps.
                    user = (await session.exec(select(User).where(User.user_id == user_id))).one()
                    if user.points < credits_to_cashout:
                        raise ValueError("User does not have enough points to cash out")

                    # 2. Create the payment transaction.
                    session.add(
                        PaymentTransaction(
                            payment_transaction_id=payment_transaction_id,
                            currency=self.currency,
                            amount=amount,
                            usd_amount=usd_amount,
                            source_instrument_id=source_instrument_id,
                            destination_instrument_id=destination_instrument_id,
                            status=PaymentTransactionStatusEnum.NOT_STARTED,
                            last_status_change_at=datetime.now(),
                            additional_info={
                                "user_id": user_id,
                                "destination_identifier": destination_identifier,
                                "destination_identifier_type": str(destination_identifier_type),
                            },
                        )
                    )
                    await session.flush()

                    # 3. Update the user's points.
                    await session.exec(
                        update(User).values(points=User.points - credits_to_cashout).where(User.user_id == user_id)  # type: ignore
                    )

                    session.add(
                        PointTransaction(
                            user_id=user_id,
                            point_delta=-credits_to_cashout,
                            action_type=PointsActionEnum.CASHOUT,
                            cashout_payment_transaction_id=payment_transaction_id,
                        )
                    )
        except Exception as e:
            log_dict = {
                "message": "Failed to initiate db for payment",
                "user_id": user_id,
                "error": str(e),
                "facilitator": self.facilitator,
                "payment_transaction_id": str(payment_transaction_id),
                "credits_to_cashout": credits_to_cashout,
                "amount": str(amount),
                "usd_amount": str(usd_amount),
                "currency": self.currency,
                "elapsed_time": time.time() - start_time,
                "destination_identifier": destination_identifier,
                "destination_identifier_type": destination_identifier_type,
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentProcessingError("Failed to initiate db for payment") from e

        db_inits_done = time.time()
        log_dict = {
            "message": "Initial DB updates made",
            "user_id": user_id,
            "processing_time": db_inits_done - balance_validation_done,
            "credits_to_cashout": credits_to_cashout,
            "amount": str(amount),
            "usd_amount": str(usd_amount),
            "currency": self.currency,
            "payment_transaction_id": str(payment_transaction_id),
            "facilitator": self.facilitator,
            "elapsed_time": time.time() - start_time,
            "destination_identifier": destination_identifier,
            "destination_identifier_type": destination_identifier_type,
        }
        logging.info(json_dumps(log_dict))

        # 4. Send the payment request to the partner.

        # Pass the destination instrument ID to create a consistent unique reference ID.
        if destination_additional_details is None:
            destination_additional_details = {}
        destination_additional_details["destination_instrument_id"] = destination_instrument_id

        if instrument_metadata is not None and "validated_vpa" in instrument_metadata:
            destination_additional_details["validated_vpa"] = instrument_metadata["validated_vpa"]

        try:
            payment_response = await self._send_payment_request(
                user_id=user_id,
                credits_to_cashout=credits_to_cashout,
                amount=amount,
                usd_amount=usd_amount,
                destination_identifier=destination_identifier,
                destination_identifier_type=destination_identifier_type,
                destination_additional_details=destination_additional_details,
                payment_transaction_id=payment_transaction_id,
            )
            payment_response_received = time.time()
            log_dict = {
                "message": ":white_check_mark: Success - Sent payment request to the partner",
                "user_id": user_id,
                "processing_time": payment_response_received - db_inits_done,
                "payment_transaction_id": str(payment_transaction_id),
                "partner_reference_id": payment_response.partner_reference_id,
                "facilitator": self.facilitator,
                "elapsed_time": time.time() - start_time,
                "credits_to_cashout": credits_to_cashout,
                "amount": str(amount),
                "usd_amount": str(usd_amount),
                "currency": self.currency,
                "destination_identifier": destination_identifier,
                "destination_identifier_type": destination_identifier_type,
            }
            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
            logging.info(json_dumps(log_dict))
        except Exception as e:
            # TODO: Have granular exceptions for different types of errors.
            # We should only reverse the transaction if the error is known to be irrecoverable
            # and that the partner isn't going to process it.

            # To be on the safe side, the code talking to the partner may have returned
            # NOT_STARTED as the status, if we do not know the status of the error.
            log_dict = {
                "message": ":x: Failure - Failed to send payment request to the partner",
                "user_id": user_id,
                "error": str(e),
                "facilitator": self.facilitator,
                "payment_transaction_id": str(payment_transaction_id),
                "elapsed_time": time.time() - start_time,
                "credits_to_cashout": credits_to_cashout,
                "amount": str(amount),
                "usd_amount": str(usd_amount),
                "currency": self.currency,
                "destination_identifier": destination_identifier,
                "destination_identifier_type": destination_identifier_type,
            }
            logging.exception(json_dumps(log_dict))
            await self.undo_payment_transaction(payment_transaction_id)
            asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
            raise PaymentProcessingError("Failed to send payment request to the partner") from e

        partner_reference_id = payment_response.partner_reference_id
        async with get_async_session() as session:
            async with session.begin():
                update_values = {
                    "status": payment_response.transaction_status,
                    "partner_reference_id": partner_reference_id,
                }
                if payment_response.customer_reference_id is not None:
                    update_values["customer_reference_id"] = payment_response.customer_reference_id

                await session.exec(
                    update(PaymentTransaction)
                    .values(**update_values)
                    .where(PaymentTransaction.payment_transaction_id == payment_transaction_id)  # type: ignore
                )
        db_updates_done = time.time()
        log_dict = {
            "message": f":white_check_mark: Success - Updated payment transaction "
            f"status to {payment_response.transaction_status}",
            "transaction_status": payment_response.transaction_status,
            "user_id": user_id,
            "processing_time": db_updates_done - payment_response_received,
            "payment_transaction_id": str(payment_transaction_id),
            "partner_reference_id": partner_reference_id,
            "facilitator": self.facilitator,
            "elapsed_time": time.time() - start_time,
            "credits_to_cashout": credits_to_cashout,
            "amount": str(amount),
            "usd_amount": str(usd_amount),
            "currency": self.currency,
            "destination_identifier": destination_identifier,
            "destination_identifier_type": destination_identifier_type,
        }
        logging.info(json_dumps(log_dict))

        # 5. Start monitoring the payment transaction and hand off to the async task.
        # TODO: Improve this.
        assert partner_reference_id is not None
        asyncio.create_task(
            self.monitor_payment_status(
                payment_transaction_id,
                partner_reference_id,
                user_id,
                current_status=payment_response.transaction_status,
            )
        )

        log_dict = {
            "message": ":white_check_mark: Success - make_payment completed",
            "user_id": user_id,
            "processing_time": time.time() - start_time,
            "payment_transaction_id": str(payment_transaction_id),
            "partner_reference_id": partner_reference_id,
            "facilitator": self.facilitator,
            "elapsed_time": time.time() - start_time,
            "credits_to_cashout": credits_to_cashout,
            "amount": str(amount),
            "usd_amount": str(usd_amount),
            "currency": self.currency,
            "destination_identifier": destination_identifier,
            "destination_identifier_type": destination_identifier_type,
        }
        logging.info(json_dumps(log_dict))

        return payment_response

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
        payment_transaction_id: uuid.UUID | None = None,
    ) -> PaymentResponse:
        assert destination_additional_details is not None
        assert payment_transaction_id is not None

        destination_upi_id = (
            destination_additional_details["validated_vpa"]
            if "validated_vpa" in destination_additional_details
            else destination_identifier
        )

        return await make_payment(
            AxisPaymentRequest(
                internal_payment_transaction_id=payment_transaction_id,
                amount=amount,
                destination_internal_id=destination_additional_details["destination_instrument_id"],
                destination_upi_id=destination_upi_id,
                # Ensure that the message is limited to 60 characters.
                # Only alphanumeric characters are allowed.
                receiver_display_message=f"{credits_to_cashout} YUPP credits redeemed"[:60],
            )
        )

    async def get_payment_status(self, payment_transaction_id: uuid.UUID) -> PaymentResponse:
        async with get_async_session() as session:
            payment_transaction = (
                await session.exec(
                    select(PaymentTransaction).where(
                        PaymentTransaction.payment_transaction_id == payment_transaction_id,
                        PaymentTransaction.deleted_at.is_(None),  # type: ignore
                    )
                )
            ).one()

        return PaymentResponse(
            payment_transaction_id=payment_transaction_id,
            transaction_status=payment_transaction.status,
            customer_reference_id=payment_transaction.customer_reference_id,
        )

    async def validate_destination_identifier(
        self, destination_identifier: str, destination_identifier_type: PaymentInstrumentIdentifierTypeEnum
    ) -> ValidateDestinationIdentifierResponse:
        verify_vpa_response = await verify_vpa(destination_identifier, destination_identifier_type)
        current_time = datetime.now(UTC)
        validation_token_expiry = current_time + timedelta(seconds=VALIDATE_DESTINATION_IDENTIFIER_TOKEN_TTL_SECONDS)
        encryptor = Fernet(settings.VALIDATE_DESTINATION_IDENTIFIER_SECRET_KEY)
        payload = {
            "input_identifier": destination_identifier,
            "input_identifier_type": str(destination_identifier_type),
            "validated_vpa": verify_vpa_response.validated_vpa,
            "customer_name": verify_vpa_response.customer_name,
            "last_validated_at": int(current_time.timestamp()),
        }
        return ValidateDestinationIdentifierResponse(
            destination_metadata=UpiDestinationMetadata(
                masked_name_from_bank=_mask_name(verify_vpa_response.customer_name),
            ),
            validated_destination_details=encryptor.encrypt_at_time(
                json.dumps(payload).encode("utf-8"),
                int(current_time.timestamp()),
            ).decode("utf-8"),
            validated_data_expiry=int(validation_token_expiry.timestamp()),
        )

    @staticmethod
    def get_polling_config() -> tuple[int, int]:
        """
        Returns polling configuration for Axis UPI payments.
        Returns:
            tuple: (max_wait_time_seconds, poll_interval_seconds)
        """
        return (
            5 * 60,  # 5 minutes in seconds
            5,  # 5 seconds
        )
