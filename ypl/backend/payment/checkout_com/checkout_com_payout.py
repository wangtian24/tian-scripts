import logging
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any, Final, Literal

import httpx
from ypl.backend.config import settings
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import CurrencyEnum

CHECKOUT_API_URL: Final[str] = settings.CHECKOUT_COM_API_URL
CHECKOUT_SECRET_KEY: Final[str] = settings.CHECKOUT_COM_SECRET
CHECKOUT_PROCESSING_CHANNEL: Final[str] = settings.CHECKOUT_COM_PROCESSING_CHANNEL

MIN_BALANCES: dict[CurrencyEnum, Decimal] = {
    CurrencyEnum.USD: Decimal(1000),
}


class TransactionStatus(StrEnum):
    """Enum for Checkout.com transaction statuses."""

    PENDING = "Pending"
    DECLINED = "Declined"
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

    reference: Literal["YUPP Payout"]


@dataclass(frozen=True)
class Instruction:
    """Represents an instruction for Checkout.com."""

    purpose: Literal["YUPP Payout"]


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
        secret_key = settings.CHECKOUT_COM_SECRET

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
            "processing_channel_id": settings.CHECKOUT_COM_PROCESSING_CHANNEL,
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
                    "error": "Checkout.com: Missing id in payment response",
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
    secret_key = settings.CHECKOUT_COM_SECRET

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


@dataclass(frozen=True)
class BillingAddress:
    """Represents a billing address for bank account holder."""

    address_line1: str
    city: str
    state: str
    zip: str
    country: str

    def to_dict(self) -> dict:
        return {
            "address_line1": self.address_line1,
            "city": self.city,
            "state": self.state,
            "zip": self.zip,
            "country": self.country,
        }


@dataclass(frozen=True)
class AccountHolder:
    """Represents an account holder for bank account."""

    type: Literal["individual"]
    first_name: str
    last_name: str
    billing_address: BillingAddress

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "billing_address": self.billing_address.to_dict(),
        }


@dataclass(frozen=True)
class BankAccountInstrument:
    """Represents a bank account instrument payload for Checkout.com."""

    type: Literal["bank_account"]
    currency: Literal["USD"]
    account_type: Literal["current"]
    account_number: str
    bank_code: str
    country: str
    account_holder: AccountHolder

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "currency": self.currency,
            "account_type": self.account_type,
            "account_number": self.account_number,
            "bank_code": self.bank_code,
            "country": self.country,
            "account_holder": self.account_holder.to_dict(),
        }


async def create_checkout_instrument(user_id: str, instrument_details: BankAccountInstrument) -> str:
    """Create a payment instrument for a user at checkout.com.

    Args:
        user_id: The ID of the user
        TODO: To refactor this to enable card type too later. Hence a lot of fields hardcoded for bank account.
        instrument_details: The instrument details (e.g. bank account details in JSON format)

    Returns:
        str: The instrument ID from Checkout.com
    """
    try:
        headers = _get_auth_header(CHECKOUT_SECRET_KEY)

        payload = instrument_details.to_dict()
        log_dict = {
            "message": "Checkout.com: Creating bank account instrument",
            "user_id": user_id,
            "instrument_details": payload,
        }
        logging.info(json_dumps(log_dict))

        async with httpx.AsyncClient() as client:
            response = await client.post(f"{CHECKOUT_API_URL}/instruments", headers=headers, json=payload)

            if response.status_code not in (200, 201):
                details = {"status_code": str(response.status_code), "response": response.text}
                raise CheckoutPayoutError("Checkout.com: Failed to create bank account instrument", details)

            data = response.json()
            instrument_id = data.get("id")

            if not instrument_id:
                details = {
                    "response_data": str(data),
                    "error": "Checkout.com: Missing instrument ID in response",
                }
                raise CheckoutPayoutError(GENERIC_ERROR_MESSAGE, details)

            log_dict = {
                "message": "Checkout.com: Successfully created bank account instrument",
                "user_id": user_id,
                "instrument_id": instrument_id,
            }
            logging.info(json_dumps(log_dict))

            return str(instrument_id)

    except Exception as e:
        details = {"error": str(e)}
        raise CheckoutPayoutError("Checkout.com: Failed to create bank account instrument", details) from e


async def get_source_instrument_balance() -> Decimal:
    """Get the balance of a specific instrument."""

    try:
        headers = _get_auth_header(CHECKOUT_SECRET_KEY)

        entity_id = settings.CHECKOUT_COM_ENTITY_ID
        log_dict = {
            "message": "Checkout.com: Getting source instrument balance",
            "entity_id": entity_id,
        }
        logging.info(json_dumps(log_dict))

        async with httpx.AsyncClient() as client:
            # TODO: Add query for currency later if we have to support other currencies
            response = await client.get(f"{CHECKOUT_API_URL}/balances/{entity_id}?query=currency:USD", headers=headers)

            if response.status_code not in (200, 201):
                details = {"status_code": str(response.status_code), "response": response.text}
                raise CheckoutPayoutError("Checkout.com: Failed to get source instrument balance", details)

            data = response.json()
            available_balance = extract_available_balance(data)

            if not available_balance:
                details = {
                    "response_data": str(data),
                    "error": "Checkout.com: Missing available balance in response",
                }
                raise CheckoutPayoutError(GENERIC_ERROR_MESSAGE, details)

            log_dict = {
                "message": "Checkout.com: Successfully retrieved source instrument balance",
                "available_balance": str(available_balance),
            }
            logging.info(json_dumps(log_dict))

            return available_balance

    except Exception as e:
        details = {"error": str(e)}
        raise CheckoutPayoutError("Checkout.com: Failed to get source instrument balance", details) from e


def extract_available_balance(response_data: dict) -> Decimal | None:
    try:
        data = response_data.get("data", [])
        if data and isinstance(data, list):
            first_entry = data[0]
            balances = first_entry.get("balances", {})
            return Decimal(balances.get("available"))
        return None
    except Exception as e:
        details = {"response_data": str(response_data), "error": str(e)}
        raise CheckoutPayoutError("Checkout.com: Failed to extract available balance", details) from e
