import uuid
from decimal import Decimal

from ypl.backend.payment.base_types import PaymentResponse
from ypl.backend.payment.facilitator import BaseFacilitator
from ypl.backend.payment.upi.axis.request_utils import get_balance
from ypl.db.payments import CurrencyEnum, PaymentInstrumentIdentifierTypeEnum, PaymentTransactionStatusEnum


class AxisUpiFacilitator(BaseFacilitator):
    async def get_balance(self, currency: CurrencyEnum) -> Decimal:
        return await get_balance()

    async def get_source_instrument_id(self) -> uuid.UUID:
        # TODO: Implement this
        return uuid.uuid4()

    async def get_destination_instrument_id(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
    ) -> uuid.UUID:
        # TODO: Implement this
        return uuid.uuid4()

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
