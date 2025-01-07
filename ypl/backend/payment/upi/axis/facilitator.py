import asyncio
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy.orm import selectinload
from sqlmodel import select, update
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack, post_to_slack_with_user_name
from ypl.backend.payment.base_types import PaymentInstrumentError, PaymentProcessingError, PaymentResponse
from ypl.backend.payment.facilitator import BaseFacilitator
from ypl.backend.payment.payout_utils import (
    SLACK_WEBHOOK_CASHOUT,
    get_destination_instrument_id,
    get_source_instrument_id,
)
from ypl.backend.payment.upi.axis.request_utils import AxisPaymentRequest, get_balance, make_payment
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


class AxisUpiFacilitator(BaseFacilitator):
    # Only used as a placeholder. This ID will not be sent to the bank during the payment request.
    SOURCE_INSTRUMENT_UPI_ID = "AXIS"

    async def get_balance(self, currency: CurrencyEnum) -> Decimal:
        return await get_balance()

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
    ) -> uuid.UUID:
        return await get_destination_instrument_id(
            PaymentInstrumentFacilitatorEnum.UPI,
            user_id,
            destination_identifier,
            destination_identifier_type,
        )

    # TODO: Move this to the base facilitator.
    async def undo_payment_transaction(self, payment_transaction_id: uuid.UUID) -> None:
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
                            "payment_transaction_id": payment_transaction_id,
                        }
                        logging.info(json_dumps(log_dict))
                        raise ValueError("Payment transaction already in the final state")
                        return

                    point_transaction = payment_transaction.credits_transaction

                    log_dict = {
                        "message": "Failed to process payout reward. Reversing transaction.",
                        "user_id": point_transaction.user_id,
                        "payment_transaction_id": str(payment_transaction_id),
                        "points_transaction_id": str(point_transaction.transaction_id),
                        "credits_to_cashout": str(point_transaction.point_delta),
                        "amount": str(payment_transaction.amount),
                        "source_instrument_id": str(payment_transaction.source_instrument_id),
                        "destination_instrument_id": str(payment_transaction.destination_instrument_id),
                        "destination_identifier": payment_transaction.destination_instrument.identifier,
                        "destination_identifier_type": payment_transaction.destination_instrument.identifier_type,
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

                    session.add(payment_transaction)

                    # 5. Create a reversal payment transaction.
                    session.add(
                        PaymentTransaction(
                            payment_transaction_id=uuid.uuid4(),
                            currency=payment_transaction.currency,
                            amount=payment_transaction.amount,
                            source_instrument_id=payment_transaction.source_instrument_id,
                            destination_instrument_id=payment_transaction.destination_instrument_id,
                            status=PaymentTransactionStatusEnum.REVERSED,
                            last_status_change_at=datetime.now(),
                        )
                    )
            log_dict = {
                "message": "Successfully reversed transaction",
                "payment_transaction_id": str(payment_transaction_id),
                "points_transaction_id": str(point_transaction.transaction_id),
                "user_id": point_transaction.user_id,
                "amount": str(payment_transaction.amount),
                "source_instrument_id": str(payment_transaction.source_instrument_id),
                "destination_instrument_id": str(payment_transaction.destination_instrument_id),
                "destination_identifier": payment_transaction.destination_instrument.identifier,
                "destination_identifier_type": payment_transaction.destination_instrument.identifier_type,
            }
            logging.info(json_dumps(log_dict))
            asyncio.create_task(
                post_to_slack_with_user_name(
                    str(point_transaction.user_id), json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT
                )
            )
        except Exception as e:
            log_dict = {
                "message": "Failed to handle failed transaction cleanup",
                "payment_transaction_id": str(payment_transaction_id),
                "error": str(e),
            }
            logging.exception(json_dumps(log_dict))
            asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

    # TODO: Move this to the base facilitator.
    async def make_payment(
        self,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        destination_additional_details: dict | None = None,
    ) -> PaymentResponse:
        start_time = time.time()
        # TODO: Get this from where the payment is initiated (i.e., from the API handler).
        payment_transaction_id = uuid.uuid4()
        try:
            # 0. Get the balance of the source instrument and verify if it is enough.
            # 1. Get the source and destination instrument IDs.
            balance, source_instrument_id, destination_instrument_id = await asyncio.gather(
                self.get_balance(self.currency),
                self.get_source_instrument_id(),
                self.get_destination_instrument_id(user_id, destination_identifier, destination_identifier_type),
            )
            if balance < amount:
                log_dict = {
                    "message": "Source instrument does not have enough balance",
                    "user_id": user_id,
                    "amount": str(amount),
                    "currency": self.currency,
                    "source_instrument_balance": str(balance),
                    "facilitator": self.facilitator,
                    "payment_transaction_id": payment_transaction_id,
                }
                logging.error(json_dumps(log_dict))
                raise ValueError("Source instrument does not have enough balance")
        except Exception as e:
            log_dict = {
                "message": "Failed to initiate make_payment",
                "user_id": user_id,
                "error": str(e),
                "facilitator": self.facilitator,
                "payment_transaction_id": payment_transaction_id,
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentInstrumentError("Failed to get payment instruments") from e
        balance_validation_done = time.time()
        log_dict = {
            "message": "Successfully got payment instruments and validated balance",
            "user_id": user_id,
            "payment_transaction_id": payment_transaction_id,
            "duration": balance_validation_done - start_time,
        }
        logging.info(json_dumps(log_dict))

        try:
            async with get_async_session() as session:
                async with session.begin():
                    # Set the isolation level to SERIALIZABLE to prevent concurrent
                    # updates of all the payment transactions.
                    await session.connection(execution_options={"isolation_level": "SERIALIZABLE"})

                    # 2. Create the payment transaction.
                    session.add(
                        PaymentTransaction(
                            payment_transaction_id=payment_transaction_id,
                            currency=self.currency,
                            amount=amount,
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
                "payment_transaction_id": payment_transaction_id,
            }
            logging.exception(json_dumps(log_dict))
            raise PaymentProcessingError("Failed to initiate db for payment") from e

        db_inits_done = time.time()
        log_dict = {
            "message": "Initial DB updates made",
            "user_id": user_id,
            "duration": db_inits_done - balance_validation_done,
            "payment_transaction_id": payment_transaction_id,
        }
        logging.info(json_dumps(log_dict))

        # 4. Send the payment request to the partner.

        # Pass the destination instrument ID to create a consistent unique reference ID.
        if destination_additional_details is None:
            destination_additional_details = {}
        destination_additional_details["destination_instrument_id"] = destination_instrument_id

        try:
            payment_response = await self._send_payment_request(
                user_id=user_id,
                credits_to_cashout=credits_to_cashout,
                amount=amount,
                destination_identifier=destination_identifier,
                destination_identifier_type=destination_identifier_type,
                destination_additional_details=destination_additional_details,
            )
            payment_response_received = time.time()
            log_dict = {
                "message": "Successfully sent payment request to the partner",
                "user_id": user_id,
                "duration": payment_response_received - db_inits_done,
                "payment_transaction_id": payment_transaction_id,
            }
            logging.info(json_dumps(log_dict))
        except Exception as e:
            log_dict = {
                "message": "Failed to send payment request to the partner",
                "user_id": user_id,
                "error": str(e),
                "facilitator": self.facilitator,
                "payment_transaction_id": payment_transaction_id,
            }
            logging.exception(json_dumps(log_dict))
            await self.undo_payment_transaction(payment_transaction_id)
            raise PaymentProcessingError("Failed to send payment request to the partner") from e

        async with get_async_session() as session:
            async with session.begin():
                await session.exec(
                    update(PaymentTransaction)
                    .values(
                        status=PaymentTransactionStatusEnum.PENDING,
                        partner_reference_id=str(payment_response.payment_transaction_id),
                    )
                    .where(PaymentTransaction.payment_transaction_id == payment_transaction_id)  # type: ignore
                )
        db_updates_done = time.time()
        log_dict = {
            "message": "Updated payment transaction status to PENDING",
            "user_id": user_id,
            "duration": db_updates_done - payment_response_received,
            "payment_transaction_id": payment_transaction_id,
        }
        logging.info(json_dumps(log_dict))

        # 5. Start monitoring the payment transaction and hand off to the async task.
        # TODO: Implement this

        return payment_response

    async def _send_payment_request(
        self,
        user_id: str,
        credits_to_cashout: int,
        amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        destination_additional_details: dict | None = None,
    ) -> PaymentResponse:
        assert destination_additional_details is not None

        return await make_payment(
            AxisPaymentRequest(
                internal_payment_transaction_id=uuid.uuid4(),
                amount=amount,
                destination_internal_id=destination_additional_details["destination_instrument_id"],
                destination_upi_id=destination_identifier,
                # TODO: Get the final message from Mouli.
                # Ensure that the message is limited to 60 characters.
                # Only alphanumeric characters are allowed.
                receiver_display_message=f"{credits_to_cashout} YUPP credits redeemed"[:60],
            )
        )

    async def get_payment_status(self, payment_reference_id: str) -> PaymentTransactionStatusEnum:
        # TODO: Implement this
        return PaymentTransactionStatusEnum.SUCCESS
