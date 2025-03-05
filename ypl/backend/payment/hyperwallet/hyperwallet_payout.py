import json
import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Any, Final
from uuid import UUID

import httpx
from ypl.backend.config import settings
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import CurrencyEnum

HYPERWALLET_API_URL: Final[str] = settings.HYPERWALLET_API_URL
HYPERWALLET_PROGRAM_TOKEN: Final[str] = settings.HYPERWALLET_PROGRAM_TOKEN
HYPERWALLET_USERNAME: Final[str] = settings.HYPERWALLET_USERNAME
HYPERWALLET_PASSWORD: Final[str] = settings.HYPERWALLET_PASSWORD

MIN_BALANCES: dict[CurrencyEnum, Decimal] = {
    CurrencyEnum.USD: Decimal(1000),
}


class TransactionStatus(StrEnum):
    """Enum for Hyperwallet transaction statuses."""

    CREATED = "CREATED"
    SCHEDULED = "SCHEDULED"
    PENDING_ACCOUNT_ACTIVATION = "PENDING_ACCOUNT_ACTIVATION"
    PENDING_ID_VERIFICATION = "PENDING_ID_VERIFICATION"
    PENDING_TAX_VERIFICATION = "PENDING_TAX_VERIFICATION"
    PENDING_TRANSFER_METHOD_ACTION = "PENDING_TRANSFER_METHOD_ACTION"
    PENDING_TRANSACTION_VERIFICATION = "PENDING_TRANSACTION_VERIFICATION"
    IN_PROGRESS = "IN_PROGRESS"
    WAITING_FOR_SUPPLEMENTAL_DATA = "WAITING_FOR_SUPPLEMENTAL_DATA"
    UNCLAIMED = "UNCLAIMED"
    FAILED = "FAILED"
    RECALLED = "RECALLED"
    RETURNED = "RETURNED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    UNKNOWN = "UNKNOWN"


GENERIC_ERROR_MESSAGE: Final[str] = "Internal error"


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> str | Any:
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


class HyperwalletPayoutError(Exception):
    """Custom exception for Hyperwallet payout related errors."""

    def __init__(self, message: str = GENERIC_ERROR_MESSAGE, details: dict[str, Any] | None = None):
        """Initialize the error with a message and optional details.

        Args:
            message: Error description
            details: Additional context about the error
        """
        super().__init__(message)
        self.details = details or {}
        # Log the error with details
        log_dict: dict[str, Any] = {"message": message, "details": self.details}
        logging.error(json_dumps(log_dict))


@dataclass(frozen=True)
class HyperwalletPayout:
    """Represents a payout request for Hyperwallet.

    Attributes:
        user_id: User ID
        amount: Amount to be paid out
        payment_transaction_id: UUID of the payment transaction
        currency: Type of currency
        destination_token: Hyperwallet transfer method token
    """

    user_id: str
    amount: Decimal
    payment_transaction_id: UUID
    currency: CurrencyEnum
    destination_token: str

    # TODO: Create methods to retrieve the account details and corresponding balance
    # No balance check is done today


def _get_basic_auth_header(username: str | None, password: str | None) -> str:
    """Generate Basic Auth header for Hyperwallet API."""
    if not username or not password:
        raise HyperwalletPayoutError(GENERIC_ERROR_MESSAGE, {"error": "Missing API credentials"})
    import base64

    credentials = f"{username}:{password}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


async def process_hyperwallet_payout(payout: HyperwalletPayout) -> tuple[str, str]:
    """Process a Hyperwallet payout.

    Returns:
        Tuple[str, str]: A tuple containing (transaction_token, transaction_status)
    """
    log_dict: dict[str, Any] = {
        "message": "Hyperwallet: Processing payout",
        "user_id": str(payout.user_id),
        "amount": str(payout.amount),
        "currency": str(payout.currency.value),
        "payment_transaction_id": str(payout.payment_transaction_id),
        "destination_token": str(payout.destination_token),
    }
    logging.info(json_dumps(log_dict))

    # Validate input values
    if not all([payout.destination_token, payout.amount, payout.currency, payout.payment_transaction_id]):
        validation_details: dict[str, Any] = {
            "message": "Hyperwallet: Missing required fields",
            "has_destination_token": bool(payout.destination_token),
            "has_amount": bool(payout.amount),
            "has_currency": bool(payout.currency),
            "has_payment_transaction_id": bool(payout.payment_transaction_id),
            "error": "Missing required fields",
        }
        raise HyperwalletPayoutError(GENERIC_ERROR_MESSAGE, validation_details)

    # TODO: Create methods to retrieve the account details and corresponding balance
    # No balance check is done today. Put an alert on slack if the balance is low.
    # if available_balance < payout.amount:
    #     balance_details: dict[str, Any] = {
    #         "user_id": str(payout.user_id),
    #         "has_sufficient_balance": False,
    #         "currency": str(payout.currency.value),
    #         "available_balance": str(available_balance),
    #         "payout_amount": str(payout.amount),
    #         "error": "Insufficient balance to make payment",
    #     }
    #     raise HyperwalletPayoutError(GENERIC_ERROR_MESSAGE, balance_details)

    # Round the amount to 2 decimal places for fiat currency
    rounded_amount = payout.amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    try:
        program_token = HYPERWALLET_PROGRAM_TOKEN
        api_username = HYPERWALLET_USERNAME
        api_password = HYPERWALLET_PASSWORD

        if not all([program_token, api_username, api_password]):
            raise HyperwalletPayoutError(
                GENERIC_ERROR_MESSAGE, {"error": "Hyperwallet: API credentials not found in environment variables"}
            )

        headers = {
            "Authorization": _get_basic_auth_header(api_username, api_password),
            "Content-Type": "application/json",
        }

        payload = {
            "amount": str(rounded_amount),
            "clientPaymentId": str(payout.payment_transaction_id),
            "currency": payout.currency.value,
            "destinationToken": payout.destination_token,
            "programToken": program_token,
            "purpose": "OTHER",
        }
        log_dict = {
            "message": "Hyperwallet: Calling Hyperwallet API with Payout payload",
            "payload": payload,
        }
        logging.info(json_dumps(log_dict))

        async with httpx.AsyncClient() as client:
            response = await client.post(f"{HYPERWALLET_API_URL}/payments", headers=headers, json=payload)

            if response.status_code not in (200, 201):
                details = {"status_code": str(response.status_code), "response": response.text}
                raise HyperwalletPayoutError(response.text, details)

            data = response.json()
            transaction_token = data.get("token")
            transaction_status = data.get("status", TransactionStatus.UNKNOWN.value)

            if not transaction_token:
                details = {
                    "response_data": str(data),
                    "error": "Hyperwallet: Missing token in payment response",
                }
                raise HyperwalletPayoutError(GENERIC_ERROR_MESSAGE, details)

            log_dict = {
                "message": "Hyperwallet payout created",
                "user_id": str(payout.user_id),
                "amount": str(payout.amount),
                "currency": str(payout.currency.value),
                "transaction_token": transaction_token,
                "transaction_status": transaction_status,
            }
            logging.info(json_dumps(log_dict))

            return transaction_token, transaction_status

    except HyperwalletPayoutError:
        raise
    except Exception as e:
        log_dict = {
            "message": "Hyperwallet: Error processing payout",
            "user_id": str(payout.user_id),
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        details = {
            "message": "Hyperwallet: Error processing payout",
            "user_id": str(payout.user_id),
            "error": str(e),
        }
        raise HyperwalletPayoutError(str(e), details) from e


async def get_transaction_status(transaction_token: str) -> str:
    """Get the status of a specific transaction.

    Args:
        transaction_token: The token of the transaction to check

    Returns:
        str: The status of the transaction
    """
    api_username = HYPERWALLET_USERNAME
    api_password = HYPERWALLET_PASSWORD

    if not all([api_username, api_password]):
        raise HyperwalletPayoutError(
            GENERIC_ERROR_MESSAGE, {"error": "Hyperwallet API credentials not found in environment variables"}
        )

    headers = {
        "Authorization": _get_basic_auth_header(api_username, api_password),
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HYPERWALLET_API_URL}/payments/{transaction_token}", headers=headers)

            if response.status_code != 200:
                details = {"status_code": str(response.status_code), "response": response.text}
                raise HyperwalletPayoutError("Hyperwallet: Failed to get transaction status", details)

            data = response.json()
            status = data.get("status", TransactionStatus.UNKNOWN.value)

            log_dict = {
                "message": "Hyperwallet: Retrieved transaction status",
                "transaction_token": transaction_token,
                "status": status,
            }
            logging.info(json_dumps(log_dict))

            return str(status)

    except Exception as e:
        details = {"transaction_token": transaction_token, "error": str(e)}
        raise HyperwalletPayoutError("Hyperwallet: Error getting payout status", details) from e
