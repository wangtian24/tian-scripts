import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final

from stripe import StripeClient
from ypl.backend.config import settings
from ypl.backend.utils.json import json_dumps

GENERIC_ERROR_MESSAGE: Final[str] = "Internal error"


@dataclass
class StripeBalance:
    """Data class representing a Stripe account balance."""

    account_id: str
    currency: str
    balance_amount: Decimal


class StripePayoutError(Exception):
    """Custom exception for Stripe payout related errors."""

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


def _get_stripe_client() -> StripeClient:
    """Get Stripe client instance."""
    try:
        stripe_config = settings.STRIPE_CONFIG
        if not stripe_config or not isinstance(stripe_config, dict):
            raise StripePayoutError(
                "Invalid Stripe configuration", {"error": "Stripe configuration is not a valid dictionary"}
            )

        secret_key = stripe_config.get("secret_key")
        if not secret_key:
            raise StripePayoutError("Missing Stripe credentials", {"error": "Stripe secret key is not configured"})

        return StripeClient(secret_key)

    except Exception as e:
        log_dict = {
            "message": "Error initializing Stripe client",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        logging.error(json_dumps(log_dict))
        raise StripePayoutError("Failed to initialize Stripe client", {"error": str(e)}) from e


async def get_stripe_balances() -> list[StripeBalance]:
    """Get the balances of a Stripe account.

    Returns:
        list[StripeBalance]: List of StripeBalance objects containing account_id, currency and balance_amount
    """
    try:
        log_dict = {"message": "Stripe: Getting account balances"}
        logging.info(json_dumps(log_dict))

        client = _get_stripe_client()
        try:
            response = client.v2.financial_accounts.list()
            if not response.data:
                log_dict = {"message": "Stripe: No financial accounts found"}
                logging.warning(json_dumps(log_dict))
                return []

            balances: list[StripeBalance] = []
            for account in response.data:
                account_id = account.id
                if hasattr(account, "balance") and hasattr(account.balance, "cash"):
                    for currency, balance_data in account.balance.cash.items():
                        currency_upper = currency.upper()
                        available_balance = Decimal(str(balance_data.value)) / 100
                        balance = StripeBalance(
                            account_id=account_id, currency=currency_upper, balance_amount=available_balance
                        )
                        balances.append(balance)

            log_dict = {
                "message": "Stripe: Retrieved account balances",
                "balances": json_dumps([vars(b) for b in balances]),
            }
            logging.info(json_dumps(log_dict))

            return balances

        except Exception as e:
            log_dict = {
                "message": "Stripe: Error getting account balances",
                "error": str(e),
                "error_type": type(e).__name__,
            }
            logging.error(json_dumps(log_dict))
            return []

    except Exception as e:
        log_dict = {"message": "Stripe: Error getting account balances", "error": str(e)}
        logging.warning(json_dumps(log_dict))
        return []
