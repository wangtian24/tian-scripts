import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID

from sqlmodel import select
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import (
    CurrencyEnum,
    PaymentInstrumentFacilitatorEnum,
    PaymentInstrumentIdentifierTypeEnum,
    PaymentTransactionStatusEnum,
)


class PaymentInstrumentError(Exception):
    """Error retrieving or creating payment instruments"""

    pass


class PaymentInstrumentNotFoundError(PaymentInstrumentError):
    """Exception raised when a payment instrument is not found."""

    pass


class PaymentDestinationIdentifierValidationError(Exception):
    """Exception raised when a payment destination identifier is not valid."""

    pass


class PaymentProcessingError(Exception):
    """Base exception for payment processing errors"""

    pass


class PaymentStatusFetchError(Exception):
    """Exception raised when a payment status fetch fails. It is a retryable error."""

    pass


class TransactionCreationError(PaymentProcessingError):
    """Error creating payment transaction"""

    pass


class PointTransactionCreationError(PaymentProcessingError):
    """Error creating point transaction"""

    pass


@dataclass
class PaymentResponse:
    # The internal transaction ID that we use to track the transaction in our system.
    payment_transaction_id: UUID
    # The status of the transaction.
    transaction_status: PaymentTransactionStatusEnum
    # The ID that the customer can use to track the transaction.
    # E.g. "transaction hash" in crypto, UTR Number in UPI.
    # May not be available immediately in all cases, so we set it only if it is available.
    customer_reference_id: str | None = None
    # The partner reference ID that we use to track the transaction in the partner's system.
    partner_reference_id: str | None = None


@dataclass
class UpiDestinationMetadata:
    masked_name_from_bank: str


@dataclass
class ValidateDestinationIdentifierResponse:
    # The encrypted token that we send to the client, which we'll get back during the payment request.
    # We will confirm that the client has validated the destination before accepting the payment request.
    validated_destination_details: str
    validated_data_expiry: int | None = None
    destination_metadata: UpiDestinationMetadata | None = None


class BaseFacilitator(ABC):
    def __init__(
        self,
        currency: CurrencyEnum,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        facilitator: PaymentInstrumentFacilitatorEnum | None = None,
    ):
        self.currency = currency
        self.destination_identifier_type = destination_identifier_type
        self.facilitator = facilitator

    # TODO(arawind): Remove payment_transaction_id after adding contextual logging.
    @abstractmethod
    async def get_balance(self, currency: CurrencyEnum, payment_transaction_id: UUID | None = None) -> Decimal:
        """Get the balance of the payment instrument.

        Args:
            currency: The currency if there are multiple currencies supported.
            payment_transaction_id: The ID of the payment transaction, if this request is part of a payment transaction.
                This is only used for logging purposes.
        """
        pass

    @abstractmethod
    async def get_source_instrument_id(self) -> UUID:
        pass

    @abstractmethod
    async def get_destination_instrument_id(
        self,
        user_id: str,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        instrument_metadata: dict | None = None,
    ) -> UUID:
        pass

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
        return await self._send_payment_request(
            user_id,
            credits_to_cashout,
            amount,
            usd_amount,
            destination_identifier,
            destination_identifier_type,
            destination_additional_details,
        )

    @abstractmethod
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
        payment_transaction_id: UUID | None = None,
    ) -> PaymentResponse:
        pass

    @abstractmethod
    async def get_payment_status(self, payment_transaction_id: uuid.UUID) -> PaymentResponse:
        pass

    # This is optional for facilitators that don't need it.
    async def validate_destination_identifier(
        self, destination_identifier: str, destination_identifier_type: PaymentInstrumentIdentifierTypeEnum
    ) -> ValidateDestinationIdentifierResponse:
        raise NotImplementedError("Method is not implemented for this facilitator")

    @staticmethod
    def init(
        currency: CurrencyEnum,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
        facilitator: PaymentInstrumentFacilitatorEnum | None = None,
        destination_additional_details: dict | None = None,
    ) -> "BaseFacilitator":
        from ypl.backend.payment.checkout_com.checkout_com_facilitator import CheckoutFacilitator
        from ypl.backend.payment.coinbase.coinbase_facilitator import CoinbaseFacilitator
        from ypl.backend.payment.facilitator import OnChainFacilitator
        from ypl.backend.payment.hyperwallet.hyperwallet_facilitator import HyperwalletFacilitator
        from ypl.backend.payment.paypal.paypal_facilitator import PayPalFacilitator
        from ypl.backend.payment.plaid.plaid_facilitator import PlaidFacilitator
        from ypl.backend.payment.upi.axis.facilitator import AxisUpiFacilitator

        if currency == CurrencyEnum.INR:
            return AxisUpiFacilitator(currency, destination_identifier_type, facilitator)
        elif currency == CurrencyEnum.USD and facilitator == PaymentInstrumentFacilitatorEnum.HYPERWALLET:
            return HyperwalletFacilitator(currency, destination_identifier_type, facilitator)
        elif currency == CurrencyEnum.USD and facilitator == PaymentInstrumentFacilitatorEnum.CHECKOUT_COM:
            return CheckoutFacilitator(currency, destination_identifier_type, facilitator)
        elif currency == CurrencyEnum.USD and facilitator == PaymentInstrumentFacilitatorEnum.PAYPAL:
            return PayPalFacilitator(currency, destination_identifier_type, facilitator)
        elif currency.is_crypto():
            if facilitator == PaymentInstrumentFacilitatorEnum.COINBASE:
                return CoinbaseFacilitator(currency, destination_identifier_type, facilitator)
            else:
                return OnChainFacilitator(currency, destination_identifier_type, facilitator)
        elif currency == CurrencyEnum.USD and facilitator == PaymentInstrumentFacilitatorEnum.PLAID:
            return PlaidFacilitator(currency, destination_identifier_type, facilitator, destination_additional_details)

        raise ValueError(f"Unsupported currency: {currency}")

    @staticmethod
    async def for_payment_transaction_id(payment_transaction_id: uuid.UUID) -> "BaseFacilitator":
        from ypl.backend.db import get_async_session
        from ypl.db.payments import (
            PaymentInstrument,
            PaymentTransaction,
        )

        log_dict = {
            "message": "Getting facilitator for payment transaction ID",
            "payment_transaction_id": str(payment_transaction_id),
        }
        logging.info(json_dumps(log_dict))
        async with get_async_session() as session:
            query = (
                select(PaymentTransaction, PaymentInstrument)
                .join(
                    PaymentInstrument,
                    PaymentTransaction.source_instrument_id == PaymentInstrument.payment_instrument_id,  # type: ignore
                )
                .where(PaymentTransaction.payment_transaction_id == payment_transaction_id)
            )
            result = await session.exec(query)
            row = result.first()
            if not row:
                raise PaymentStatusFetchError(f"Payment transaction {payment_transaction_id} not found")

            transaction: PaymentTransaction = row[0]
            source_instrument: PaymentInstrument = row[1]

            return BaseFacilitator.init(
                currency=transaction.currency,
                destination_identifier_type=source_instrument.identifier_type,
                facilitator=source_instrument.facilitator,
            )
