import enum
import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Column, Numeric
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel
from ypl.db.users import User

if TYPE_CHECKING:
    from ypl.db.point_transactions import PointTransaction


class CurrencyEnum(enum.Enum):
    INR = "INR"
    USD = "USD"
    USDC = "USDC"
    ETH = "ETH"
    BTC = "BTC"
    CBBTC = "CBBTC"

    # Keep these updated whenever a new crypto currency is added above.
    def is_crypto(self) -> bool:
        return self in [CurrencyEnum.ETH, CurrencyEnum.BTC, CurrencyEnum.USDC, CurrencyEnum.CBBTC]


class PaymentInstrumentFacilitatorEnum(enum.Enum):
    BANK = "bank"
    COINBASE = "coinbase"
    PAYPAL = "paypal"
    ON_CHAIN = "on_chain"
    BINANCE = "binance"
    CRYPTO_COM = "crypto_com"
    UPI = "upi"
    PLAID = "plaid"
    STRIPE = "stripe"
    HYPERWALLET = "hyperwallet"
    CHECKOUT_COM = "checkout.com"
    TABAPAY = "tabapay"


class PaymentInstrumentIdentifierTypeEnum(enum.Enum):
    UPI_ID = "upi_id"
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    CRYPTO_ADDRESS = "crypto_address"
    BANK_ACCOUNT_NUMBER = "bank_account_number"
    PARTNER_IDENTIFIER = "partner_identifier"
    PAYPAL_ID = "paypal_id"
    VENMO_ID = "venmo_id"


class PaymentInstrument(BaseModel, table=True):
    __tablename__ = "payment_instruments"

    payment_instrument_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    user_id: str = Field(foreign_key="users.user_id", nullable=False, index=True)
    user: "User" = Relationship(back_populates="payment_instruments")

    # The facilitator involved in the payment.
    facilitator: PaymentInstrumentFacilitatorEnum = Field(nullable=False)

    identifier_type: PaymentInstrumentIdentifierTypeEnum = Field(nullable=False)
    identifier: str = Field(nullable=False)

    # JSON metadata for storing additional instrument-specific data
    instrument_metadata: dict | None = Field(sa_column=Column(JSONB, nullable=True))

    source_transactions: list["PaymentTransaction"] = Relationship(
        back_populates="source_instrument",
        sa_relationship_kwargs={"foreign_keys": "[PaymentTransaction.source_instrument_id]"},
    )

    destination_transactions: list["PaymentTransaction"] = Relationship(
        back_populates="destination_instrument",
        sa_relationship_kwargs={"foreign_keys": "[PaymentTransaction.destination_instrument_id]"},
    )

    daily_balances: list["DailyAccountBalanceHistory"] = Relationship(back_populates="payment_instrument")


class PaymentTransactionStatusEnum(enum.Enum):
    # We just have the row in the database, nothing has been initiated yet.
    NOT_STARTED = "not_started"

    # Transaction has been initiated, but not yet completed.
    # It will remain in this state until the facilitator confirms the success or failure.
    PENDING = "pending"

    # Facilitator has confirmed the transaction was successful.
    SUCCESS = "success"

    # Facilitator has confirmed the transaction has failed.
    FAILED = "failed"

    # The transaction has been reversed.
    REVERSED = "reversed"


class PaymentTransaction(BaseModel, table=True):
    __tablename__ = "payment_transactions"

    payment_transaction_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)

    source_instrument_id: uuid.UUID = Field(foreign_key="payment_instruments.payment_instrument_id", nullable=False)
    source_instrument: "PaymentInstrument" = Relationship(
        back_populates="source_transactions",
        sa_relationship_kwargs={"foreign_keys": "[PaymentTransaction.source_instrument_id]"},
    )

    destination_instrument_id: uuid.UUID = Field(
        foreign_key="payment_instruments.payment_instrument_id", nullable=False, index=True
    )
    destination_instrument: "PaymentInstrument" = Relationship(
        back_populates="destination_transactions",
        sa_relationship_kwargs={"foreign_keys": "[PaymentTransaction.destination_instrument_id]"},
    )

    currency: CurrencyEnum = Field(nullable=False)
    # The amount of the transaction in the currency.
    # Amount is always positive.
    amount: Decimal = Field(sa_column=Column(Numeric(precision=38, scale=18), nullable=False))

    # The USD equivalent amount of the transaction at the time it was created
    usd_amount: Decimal = Field(sa_column=Column(Numeric(precision=12, scale=2), nullable=True))

    status: PaymentTransactionStatusEnum = Field(
        sa_column=Column(
            SQLAlchemyEnum(PaymentTransactionStatusEnum),
            nullable=False,
            default=PaymentTransactionStatusEnum.NOT_STARTED,
            server_default=PaymentTransactionStatusEnum.NOT_STARTED.name,
        )
    )
    last_status_change_at: datetime = Field(nullable=False)

    # The reference id from the partner's system, used to track the status of the transaction.
    # Only set after the transaction is successfully initiated.
    # This is used to track the transaction on the partner's system by our system.
    # It may not be the same as the customer_reference_id.
    partner_reference_id: str = Field(nullable=True)

    # Set if this transaction is associated with a cashout.
    credits_transaction: "PointTransaction" = Relationship(back_populates="cashout_payment_transaction")

    # Additional info for the transaction that is not part of the core schema.
    additional_info: dict | None = Field(sa_column=Column(JSONB, nullable=True))

    # The customer reference id from the partner's system. User can use this to track the transaction on
    # partner's system too.
    # E.g. for UPI this is the UTR number.
    customer_reference_id: str | None = Field(nullable=True)


class DailyAccountBalanceHistory(BaseModel, table=True):
    __tablename__ = "daily_account_balance_history"

    daily_account_balance_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    payment_instrument_id: uuid.UUID = Field(
        foreign_key="payment_instruments.payment_instrument_id", nullable=False, index=True
    )
    payment_instrument: "PaymentInstrument" = Relationship(back_populates="daily_balances")
    account_id: str = Field(nullable=False)
    currency: CurrencyEnum = Field(nullable=False)
    balance: Decimal = Field(sa_column=Column(Numeric(precision=30, scale=18), nullable=False))
