import uuid
from dataclasses import dataclass
from decimal import Decimal


@dataclass
class GetBalanceRequest:
    request_id: str
    # This will be set if the request is being made during a payment transaction.
    internal_payment_transaction_id: uuid.UUID | None = None


@dataclass
class GetBalanceResponse:
    balance: Decimal
    ip_address: str
