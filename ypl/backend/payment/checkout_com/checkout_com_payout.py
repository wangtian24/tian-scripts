import logging
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any, Final, Literal

import httpx
from ypl.backend.config import settings
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import CurrencyEnum

CHECKOUT_API_URL: Final[str] = settings.checkout_com_api_url
CHECKOUT_SECRET_KEY: Final[str] = settings.checkout_com_secret
CHECKOUT_PROCESSING_CHANNEL: Final[str] = settings.checkout_com_processing_channel

MIN_BALANCES: dict[CurrencyEnum, Decimal] = {
    CurrencyEnum.USD: Decimal(1000),
}


class TransactionStatus(StrEnum):
    """Enum for Checkout.com transaction statuses."""

    PENDING = "Pending"
    RETURNED = "Returned"
    PAID = "Paid"


@dataclass(frozen=True)
class CurrencyAccountSource:
    """Represents a currency account source for Checkout.com."""

    type: Literal["currency_account"]
    id: str


@dataclass(frozen=True)
class IdDestination:
    """Represents a destination by ID for Checkout.com."""

    type: Literal["id"]
    id: str


@dataclass(frozen=True)
class BillingDescriptor:
    """Represents a billing descriptor for Checkout.com."""

    reference: str = "YUPP Payout"


@dataclass(frozen=True)
class Instruction:
    """Represents an instruction for Checkout.com."""

    purpose: str = "YUPP Payout"


@dataclass(frozen=True)
class CheckoutPayout:
    """Represents a payout request for Checkout.com.

    Attributes:
        source: The source currency account
        destination: The destination ID
        amount: USD in cents. This will be divided by 100
        currency: Type of currency
        reference: payment_transaction_id for us to track the payout
        billing_descriptor: Details about the billing descriptor
        instruction: Additional instructions for the payout
    """

    source: CurrencyAccountSource
    destination: IdDestination
    amount: Decimal
    currency: CurrencyEnum
    reference: str
    billing_descriptor: BillingDescriptor
    instruction: Instruction


GENERIC_ERROR_MESSAGE: Final[str] = "Internal error"


class CheckoutPayoutError(Exception):
    """Custom exception for Checkout.com payout related errors."""

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


def _get_auth_header(secret_key: str | None) -> dict[str, str]:
    """Generate Authorization header for Checkout.com API."""
    if not secret_key:
        raise CheckoutPayoutError(GENERIC_ERROR_MESSAGE, {"error": "Missing API credentials"})
    return {
        "Authorization": f"Bearer {secret_key}",
        "Content-Type": "application/json",
    }


async def process_checkout_payout(payout: CheckoutPayout) -> tuple[str, str]:
    """Process a Checkout.com payout.

    Returns:
        Tuple[str, str]: A tuple containing (transaction_token, transaction_status)
    """
    log_dict: dict[str, Any] = {
        "message": "Checkout.com: Processing payout",
        "amount": str(payout.amount),
        "currency": str(payout.currency.value),
        "reference": payout.reference,
        "source": payout.source,
        "destination": payout.destination,
        "instruction": payout.instruction,
    }
    logging.info(json_dumps(log_dict))

    # Validate input values
    if not all([payout.amount, payout.currency, payout.reference, payout.source, payout.destination]):
        validation_details: dict[str, Any] = {
            "message": "Checkout.com: Missing required fields",
            "has_amount": bool(payout.amount),
            "has_currency": bool(payout.currency),
            "has_reference": bool(payout.reference),
            "has_source": bool(payout.source),
            "has_destination": bool(payout.destination),
            "error": "Missing required fields",
        }
        raise CheckoutPayoutError(GENERIC_ERROR_MESSAGE, validation_details)

    # checkout.com requires the amount to be in cents
    if payout.currency != CurrencyEnum.USD:
        raise CheckoutPayoutError(GENERIC_ERROR_MESSAGE, {"error": "Checkout.com: Only USD is supported"})
    rounded_amount = int(payout.amount * 100)

    try:
        secret_key = settings.checkout_com_secret

        if not secret_key:
            raise CheckoutPayoutError(
                GENERIC_ERROR_MESSAGE, {"error": "Checkout.com: API credentials not found in environment variables"}
            )

        headers = _get_auth_header(secret_key)

        payload = {
            "source": {
                "type": "currency_account",
                "id": payout.source.id,
            },
            "destination": {
                "type": "id",
                "id": payout.destination.id,
            },
            "amount": rounded_amount,
            "currency": payout.currency.value,
            "reference": payout.reference,
            "billing_descriptor": {
                "reference": payout.billing_descriptor.reference,
            },
            "instruction": {
                "purpose": payout.instruction.purpose,
            },
            "processing_channel_id": settings.checkout_com_processing_channel,
        }

        log_dict = {
            "message": "Checkout.com: Calling Checkout.com API with Payout payload",
            "payload": payload,
        }
        logging.info(json_dumps(log_dict))

        async with httpx.AsyncClient() as client:
            response = await client.post(f"{CHECKOUT_API_URL}/payments", headers=headers, json=payload)

            if response.status_code not in (200, 201, 202):
                details = {"status_code": str(response.status_code), "response": response.text}
                raise CheckoutPayoutError(response.text, details)

            data = response.json()
            transaction_token = data.get("id")
            transaction_status = data.get("status", TransactionStatus.PENDING.value)

            if not transaction_token:
                details = {
                    "response_data": str(data),
                    "error": "Checkout.com: Missing token in payment response",
                }
                raise CheckoutPayoutError(GENERIC_ERROR_MESSAGE, details)

            log_dict = {
                "message": "Checkout.com payout created",
                "amount": str(payout.amount),
                "currency": str(payout.currency.value),
                "transaction_token": transaction_token,
                "transaction_status": transaction_status,
                "transaction": data,
            }
            logging.info(json_dumps(log_dict))

            return transaction_token, transaction_status

    except CheckoutPayoutError as e:
        log_dict = {
            "message": "Checkout.com: Error from checkout.com processing payout",
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise
    except Exception as e:
        log_dict = {
            "message": "Checkout.com: Error while processing payout",
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise


async def get_transaction_status(transaction_token: str) -> str:
    """Get the status of a specific transaction.

    Args:
        transaction_token: The token of the transaction to check

    Returns:
        str: The status of the transaction
    """
    secret_key = settings.checkout_com_secret

    if not secret_key:
        raise CheckoutPayoutError(
            GENERIC_ERROR_MESSAGE, {"error": "Checkout.com API credentials not found in environment variables"}
        )

    headers = _get_auth_header(secret_key)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{CHECKOUT_API_URL}/payments/{transaction_token}", headers=headers)

            if response.status_code != 200:
                details = {"status_code": str(response.status_code), "response": response.text}
                raise CheckoutPayoutError("Checkout.com: Failed to get transaction status", details)

            data = response.json()
            status = data.get("status", TransactionStatus.PENDING.value)

            log_dict = {
                "message": "Checkout.com: Retrieved transaction status",
                "transaction_token": transaction_token,
                "status": status,
            }
            logging.info(json_dumps(log_dict))

            return str(status)

    except Exception as e:
        details = {"transaction_token": transaction_token, "error": str(e)}
        raise CheckoutPayoutError("Checkout.com: Error getting payout status", details) from e


async def create_checkout_instrument(user_id: str, destination_identifier: str) -> str:
    """Create a checkout instrument for a user.

    Args:
        user_id: The ID of the user
        destination_identifier: The destination identifier (e.g. wallet address)

    Returns:
        str: The instrument ID
    """
    # TODO: Implement this
    return ""
