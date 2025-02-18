import asyncio
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Any, Final
from uuid import UUID

import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import CurrencyEnum

API_VERSION: Final[str] = "v2"
BASE_URL: Final[str] = f"https://api.coinbase.com/{API_VERSION}"
REQUEST_HOST: Final[str] = "api.coinbase.com"
ENVIRONMENT: Final[str] = os.getenv("ENVIRONMENT", "staging")

MIN_BALANCES: dict[CurrencyEnum, Decimal] = {
    CurrencyEnum.ETH: Decimal(0.25),
    CurrencyEnum.USDC: Decimal(200),
}


class TransactionStatus(StrEnum):
    """Enum for Coinbase transaction statuses."""

    CANCELED = "canceled"
    COMPLETED = "completed"
    EXPIRED = "expired"
    FAILED = "failed"
    PENDING = "pending"
    WAITING_FOR_CLEARING = "waiting_for_clearing"
    WAITING_FOR_SIGNATURE = "waiting_for_signature"
    UNKNOWN = "unknown"


# only use base network for now
CRYPTO_NETWORKS: Final[dict[str, str]] = {
    CurrencyEnum.ETH.value: "base",
    CurrencyEnum.USDC.value: "base",
}

GENERIC_ERROR_MESSAGE: Final[str] = "Internal error"


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> str | Any:
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


class CoinbaseRetailPayoutError(Exception):
    """Custom exception for Coinbase retail payout related errors."""

    def __init__(self, message: str = GENERIC_ERROR_MESSAGE, details: dict[str, Any] | None = None):
        """Initialize the error with a message and optional details.

        Args:
            message: Error description
            details: Additional context about the error
        """
        super().__init__(message)
        self.details = details or {}
        # Log the error with details
        log_dict = {"message": message, "details": self.details}
        logging.error(json_dumps(log_dict))


def get_network_for_currency(currency: str) -> str | None:
    """Get the network name for a given currency.

    Args:
        currency: The currency code (e.g., 'BTC', 'ETH')

    Returns:
        Optional[str]: The network name or None if currency not supported
    """
    try:
        return CRYPTO_NETWORKS[currency]
    except Exception as e:
        log_dict = {
            "message": "Use default network for currency",
            "currency": currency,
            "error": str(e),
        }
        logging.warning(json_dumps(log_dict))
        return None


@dataclass(frozen=True)
class CoinbaseRetailPayout:
    """Represents a payout request for Coinbase retail.

    Attributes:
        user_id: Unique identifier of the user
        amount: Amount to be paid out
        to_address: Destination crypto wallet address
        currency: Type of cryptocurrency
    """

    user_id: str
    amount: Decimal
    to_address: str
    currency: CurrencyEnum
    payment_transaction_id: UUID


def build_jwt(method: str, path: str, key_name: str, key_secret: str) -> str:
    """Build a JWT token for Coinbase API authentication."""
    # Handle both \n string literals and actual newlines
    if "\\n" in key_secret:
        # If the key contains string literal \n, evaluate them
        key_secret = key_secret.encode("utf-8").decode("unicode_escape")

    # Debug log to see the exact format
    log_dict = {
        "message": "Debug PEM key format",
        "key_length": len(key_secret),
        "key_start": key_secret[:50],  # Log just the start to avoid exposing the full key
        "contains_literal_newlines": "\\n" in key_secret,
        "contains_actual_newlines": "\n" in key_secret,
    }
    logging.info(json_dumps(log_dict))

    # Ensure the key has proper line endings
    key_lines = key_secret.strip().split("\n")
    key_secret = "\n".join(key_lines) + "\n"  # Ensure there's exactly one newline at the end

    # Load and validate the private key
    private_key = serialization.load_pem_private_key(key_secret.encode("utf-8"), password=None)

    # Convert back to PEM format for JWT signing
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    uri = f"{method} {REQUEST_HOST}{path}"
    jwt_payload = {
        "sub": key_name,
        "iss": "cdp",
        "nbf": int(time.time()),
        "exp": int(time.time()) + 120,
        "uri": uri,
    }

    jwt_token = jwt.encode(
        jwt_payload,
        private_key_pem,
        algorithm="ES256",
        headers={"kid": key_name, "nonce": secrets.token_hex()},
    )
    return jwt_token


async def get_coinbase_retail_wallet_account_details() -> dict[str, dict[str, str | Decimal]]:
    """Get the account details of a Coinbase wallet.

    Returns:
        dict[str, dict[str, str | Decimal]]: A dictionary mapping currency codes to their IDs and balances
    """
    key_name = os.getenv("COINBASE_RETAIL_API_KEY_NAME")
    key_secret = os.getenv("COINBASE_RETAIL_API_SECRET")

    if not key_name or not key_secret:
        raise CoinbaseRetailPayoutError(
            "Internal error", details={"error": "Coinbase API credentials not found in environment variables"}
        )

    request_path = f"/{API_VERSION}/accounts"
    jwt_token = build_jwt("GET", request_path, key_name, key_secret)

    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/accounts", headers=headers)
        if response.status_code != 200:
            details = {"status_code": str(response.status_code), "response": response.text}
            raise CoinbaseRetailPayoutError(GENERIC_ERROR_MESSAGE, details)

        data = response.json()

        # Process the response to get balances for different accounts
        accounts: dict[str, dict[str, str | Decimal]] = {}
        for account in data.get("data", []):
            id = account.get("id")
            currency = account.get("currency", {}).get("code")
            balance = account.get("balance", {}).get("amount")
            if currency and balance:
                accounts[currency] = {"account_id": id or "", "balance": Decimal(balance) if balance else Decimal(0)}

        # Print formatted JSON
        log_dict = {
            "message": "Coinbase account balances",
            "accounts": accounts,
        }
        logging.info(json_dumps(log_dict))
        return accounts


async def get_coinbase_retail_wallet_balance_for_currency(currency: CurrencyEnum) -> dict[str, str | Decimal]:
    """Get the balance and account ID for a specific currency.

    Args:
        currency: The currency to get balance for

    Returns:
        dict[str, str | Decimal]: Dictionary containing account_id (str) and balance (Decimal)
    """
    accounts = await get_coinbase_retail_wallet_account_details()
    account_info = accounts.get(currency.value, {})

    # Convert balance to Decimal if it's a string or None
    balance = account_info.get("balance")
    if isinstance(balance, str):
        balance = Decimal(balance)
    elif not isinstance(balance, Decimal):
        balance = Decimal(0)

    # Ensure account_id is a string
    account_id = account_info.get("account_id")
    if not isinstance(account_id, str):
        account_id = ""

    return {"account_id": account_id, "balance": balance}


async def process_coinbase_retail_payout(payout: CoinbaseRetailPayout) -> tuple[str, str, str]:
    """Process a Coinbase retail payout.

    Returns:
        Tuple[str, str]: A tuple containing (transaction_id, transaction_status)
    """
    log_dict = {
        "message": "Processing Coinbase retail payout",
        "user_id": str(payout.user_id),
        "amount": str(payout.amount),
        "to_address": str(payout.to_address),
        "currency": str(payout.currency.value),
        "payment_transaction_id": str(payout.payment_transaction_id),
    }
    logging.info(json_dumps(log_dict))

    # Validate input values
    if not all([payout.user_id, payout.amount, payout.to_address, payout.currency, payout.payment_transaction_id]):
        validation_details: dict[str, Any] = {
            "has_user_id": bool(payout.user_id),
            "has_amount": bool(payout.amount),
            "has_to_address": bool(payout.to_address),
            "has_currency": bool(payout.currency),
            "has_payment_transaction_id": bool(payout.payment_transaction_id),
            "error": "Missing required fields",
        }
        raise CoinbaseRetailPayoutError(GENERIC_ERROR_MESSAGE, validation_details)

    account_info = await get_coinbase_retail_wallet_balance_for_currency(payout.currency)
    available_balance = account_info["balance"]
    account_id = str(account_info["account_id"])

    if not account_id:
        account_details: dict[str, Any] = {
            "has_account": False,
            "currency": str(payout.currency.value),
            "error": "Account not found for currency",
        }
        raise CoinbaseRetailPayoutError(GENERIC_ERROR_MESSAGE, account_details)

    if isinstance(available_balance, str):
        available_balance = Decimal(available_balance)

    min_balance = MIN_BALANCES.get(payout.currency, Decimal("0"))
    if available_balance < payout.amount + min_balance:
        message = (
            f":red_circle: *Low Balance Alert - Coinbase Retail Account*\n"
            f"Asset: {payout.currency.value}\n"
            f"Coinbase RetailAccount ID: {account_id}\n"
            f"Current Balance: {available_balance}\n"
            f"Current Transaction Required: {payout.amount}\n"
            f"Minimum Required: {min_balance}"
        )
        asyncio.create_task(post_to_slack(message))

    if available_balance < payout.amount:
        balance_details: dict[str, Any] = {
            "user_id": str(payout.user_id),
            "has_sufficient_balance": False,
            "currency": str(payout.currency.value),
            "available_balance": str(available_balance),
            "payout_amount": str(payout.amount),
            "error": "Insufficient balance to make payment",
        }
        raise CoinbaseRetailPayoutError(GENERIC_ERROR_MESSAGE, balance_details)

    # Round the amount to 8 decimal places for cryptocurrency
    rounded_amount = payout.amount.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)

    try:
        # Create the transaction
        log_dict = {
            "message": "Coinbase retail payout transaction creation called with parameters",
            "user_id": str(payout.user_id),
            "account_id": account_id,
            "to_address": str(payout.to_address),
            "amount": str(rounded_amount),
            "currency": str(payout.currency.value),
            "payment_transaction_id": str(payout.payment_transaction_id),
        }
        logging.info(json_dumps(log_dict))
        transaction = await create_transaction(
            account_id=account_id,
            to_address=payout.to_address,
            amount=str(rounded_amount),
            currency=payout.currency.value,
            payment_transaction_id=payout.payment_transaction_id,
        )

        transaction_id = transaction["id"]
        transaction_status = transaction["status"]

        log_dict = {
            "message": "Coinbase retail payout created",
            "user_id": str(payout.user_id),
            "amount": str(payout.amount),
            "transaction_id": transaction_id,
            "transaction_status": transaction_status,
        }
        logging.info(json_dumps(log_dict))

        return account_id, transaction_id, transaction_status

    except Exception as e:
        details = {
            "user_id": str(payout.user_id),
            "error": str(e),
        }
        raise CoinbaseRetailPayoutError(str(e), details) from e


async def create_transaction(
    account_id: str,
    to_address: str,
    amount: str,
    currency: str,
    payment_transaction_id: UUID,
) -> dict[str, str]:
    """Create a transaction using Coinbase's API with JWT authentication.

    Args:
        account_id: The source account ID
        to_address: The recipient's wallet address
        amount: The amount to send
        currency: The cryptocurrency to send
        payment_transaction_id: The UUID of the payment transaction

    Returns:
        dict[str, str]: Transaction details containing 'id' and 'status' fields

    Raises:
        CoinbaseRetailPayoutError: If required fields are missing from the response or if transaction creation fails
    """
    key_name = os.getenv("COINBASE_RETAIL_API_KEY_NAME")
    key_secret = os.getenv("COINBASE_RETAIL_API_SECRET")

    if not key_name or not key_secret:
        raise CoinbaseRetailPayoutError(
            GENERIC_ERROR_MESSAGE, {"error": "Coinbase API credentials not found in environment variables"}
        )

    request_path = f"/{API_VERSION}/accounts/{account_id}/transactions"
    jwt_token = build_jwt("POST", request_path, key_name, key_secret)

    # Create transaction payload
    payload = {
        "type": "send",
        "to": to_address,
        "amount": amount,
        "currency": currency,
        "idem": str(payment_transaction_id),
    }

    # Get the preferred network for the currency if available
    network = get_network_for_currency(currency)
    if network:
        payload["network"] = network

    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/accounts/{account_id}/transactions", headers=headers, json=payload
            )

            if response.status_code not in (200, 201):
                details = {"status_code": str(response.status_code), "response": response.text}
                raise CoinbaseRetailPayoutError(response.text, details)

            data = response.json()
            transaction_data = data.get("data", {})

            # Ensure we have the required fields and they are strings
            if not isinstance(transaction_data.get("id"), str) or not isinstance(transaction_data.get("status"), str):
                details = {
                    "transaction_data": str(transaction_data),
                    "error": "Missing or invalid id/status in transaction response",
                }
                raise CoinbaseRetailPayoutError(GENERIC_ERROR_MESSAGE, details)

            return {"id": transaction_data["id"], "status": transaction_data["status"]}

    except Exception as e:
        log_dict = {
            "message": ":x: *Coinbase Retail: Exception while creating transaction \n"
            + "returning blank id and pending status - Check if transaction was created on Coinbase side*",
            "to_address": str(to_address),
            "amount": str(amount),
            "currency": str(currency),
            "network": str(network),
            "error": str(e),
        }
        logging.warning(json_dumps(log_dict))
        asyncio.create_task(post_to_slack(json_dumps(log_dict)))
        return {"id": "", "status": TransactionStatus.PENDING.value}


async def get_transaction_status(account_id: str, transaction_id: str) -> str:
    """Get the status of a specific transaction.

    Args:
        account_id: The ID of the account that made the transaction
        transaction_id: The ID of the transaction to check

    Returns:
        str: The status of the transaction, one of: "pending", "completed", "failed", or "unknown"
    """
    key_name = os.getenv("COINBASE_RETAIL_API_KEY_NAME")
    key_secret = os.getenv("COINBASE_RETAIL_API_SECRET")

    if not key_name or not key_secret:
        raise CoinbaseRetailPayoutError(
            GENERIC_ERROR_MESSAGE, {"error": "Coinbase API credentials not found in environment variables"}
        )

    request_path = f"/{API_VERSION}/accounts/{account_id}/transactions/{transaction_id}"
    jwt_token = build_jwt("GET", request_path, key_name, key_secret)

    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }

    try:
        log_dict = {
            "message": "Coinbase retail payout transaction status polling for transaction",
            "account_id": account_id,
            "transaction_id": transaction_id,
        }
        logging.info(json_dumps(log_dict))
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/accounts/{account_id}/transactions/{transaction_id}", headers=headers
            )
            if response.status_code != 200:
                log_dict = {
                    "message": "Failed to get transaction status",
                    "status_code": str(response.status_code),
                    "response": response.text,
                }
                logging.error(json_dumps(log_dict))
                raise CoinbaseRetailPayoutError("Failed to get transaction status", log_dict)

            data = response.json()
            raw_status = data.get("data", {}).get("status", "")

            # Map the raw status to our known status values
            if raw_status == TransactionStatus.COMPLETED.value:
                return TransactionStatus.COMPLETED.value
            elif raw_status == TransactionStatus.FAILED.value:
                return TransactionStatus.FAILED.value
            elif raw_status == TransactionStatus.PENDING.value:
                return TransactionStatus.PENDING.value
            else:
                return TransactionStatus.UNKNOWN.value

    except Exception as e:
        details = {"transaction_id": str(transaction_id), "error": str(e)}
        raise CoinbaseRetailPayoutError("Error getting Coinbase retail payout status", details) from e
