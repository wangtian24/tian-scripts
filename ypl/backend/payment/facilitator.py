from abc import ABC, abstractmethod
from decimal import Decimal

from ypl.db.payments import CurrencyEnum, PaymentInstrumentIdentifierTypeEnum, PaymentTransactionStatusEnum


class Facilitator(ABC):
    def __init__(self, currency: CurrencyEnum):
        self.currency = currency

    @abstractmethod
    async def get_balance(self, currency: CurrencyEnum) -> Decimal:
        pass

    @abstractmethod
    async def send_payment_request(
        self,
        amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
    ) -> str:
        pass

    @abstractmethod
    async def get_payment_status(self, payment_reference_id: str) -> PaymentTransactionStatusEnum:
        pass

    async def make_payment(
        self,
        amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
    ) -> str:
        # TODO: Implement this
        # 0. Check the balance of the source account.
        # 1. Send the request to the facilitator.
        # 2. Return the transaction reference id.
        return "1234567890"

    @staticmethod
    def init(currency: CurrencyEnum, destination_identifier_type: PaymentInstrumentIdentifierTypeEnum) -> "Facilitator":
        if currency == CurrencyEnum.INR:
            return UpiFacilitator(currency)
        elif currency in (CurrencyEnum.USDC, CurrencyEnum.BTC, CurrencyEnum.ETH):
            if destination_identifier_type == PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS:
                return OnChainFacilitator(currency)
            else:
                return CoinbaseFacilitator(currency)

        raise ValueError(f"Unsupported currency: {currency}")

    @staticmethod
    async def for_transaction_reference_id(transaction_reference_id: str) -> "Facilitator":
        # TODO: Implement this
        # 1. Fetch the transaction details from the db.
        # 2. Return the facilitator for the transaction.
        return UpiFacilitator(CurrencyEnum.INR)


class UpiFacilitator(Facilitator):
    async def get_balance(self, currency: CurrencyEnum) -> Decimal:
        # TODO: Implement this
        return Decimal(1000)

    async def send_payment_request(
        self,
        amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
    ) -> str:
        # TODO: Implement this
        return "1234567890"

    async def get_payment_status(self, payment_reference_id: str) -> PaymentTransactionStatusEnum:
        # TODO: Implement this
        return PaymentTransactionStatusEnum.SUCCESS


class OnChainFacilitator(Facilitator):
    async def get_balance(self, currency: CurrencyEnum) -> Decimal:
        # TODO: Implement this
        return Decimal(1000)

    async def send_payment_request(
        self,
        amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
    ) -> str:
        # TODO: Implement this
        return "1234567890"

    async def get_payment_status(self, payment_reference_id: str) -> PaymentTransactionStatusEnum:
        # TODO: Implement this
        return PaymentTransactionStatusEnum.SUCCESS


class CoinbaseFacilitator(Facilitator):
    async def get_balance(self, currency: CurrencyEnum) -> Decimal:
        # TODO: Implement this
        return Decimal(1000)

    async def send_payment_request(
        self,
        amount: Decimal,
        destination_identifier: str,
        destination_identifier_type: PaymentInstrumentIdentifierTypeEnum,
    ) -> str:
        # TODO: Implement this
        return "1234567890"

    async def get_payment_status(self, payment_reference_id: str) -> PaymentTransactionStatusEnum:
        # TODO: Implement this
        return PaymentTransactionStatusEnum.SUCCESS
