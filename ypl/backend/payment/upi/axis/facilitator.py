import asyncio
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal

from sqlmodel import update
from ypl.backend.db import get_async_session
from ypl.backend.payment.base_types import PaymentInstrumentError, PaymentProcessingError, PaymentResponse
from ypl.backend.payment.facilitator import BaseFacilitator
from ypl.backend.payment.payout_utils import get_destination_instrument_id, get_source_instrument_id
from ypl.backend.payment.upi.axis.request_utils import get_balance
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
                                "destination_identifier_type": destination_identifier_type,
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
            "message": "Payment request completed",
            "user_id": user_id,
            "duration": payment_response_received - db_inits_done,
            "payment_transaction_id": payment_transaction_id,
        }
        logging.info(json_dumps(log_dict))

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
        # TODO: Implement this
        return PaymentResponse(
            payment_transaction_id=uuid.uuid4(),
            transaction_status=PaymentTransactionStatusEnum.SUCCESS,
            customer_reference_id="1234567890",
        )

    async def get_payment_status(self, payment_reference_id: str) -> PaymentTransactionStatusEnum:
        # TODO: Implement this
        return PaymentTransactionStatusEnum.SUCCESS
