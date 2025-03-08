import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final, cast

from stripe import StripeClient
from stripe._request_options import RequestOptions
from stripe.v2._account_service import AccountService
from stripe.v2._outbound_payment_service import OutboundPaymentService
from stripe.v2.payment_methods._us_bank_account_service import UsBankAccountService
from ypl.backend.config import settings
from ypl.backend.utils.json import json_dumps

GENERIC_ERROR_MESSAGE: Final[str] = "Internal error"


@dataclass
class StripeBalance:
    """Data class representing a Stripe account balance."""

    account_id: str
    currency: str
    balance_amount: Decimal


@dataclass
class StripeRecipientCreateRequest:
    """Data class representing a Stripe recipient account."""

    given_name: str
    surname: str
    email: str
    country: str


@dataclass
class StripeUSBankAccountCreateRequest:
    """Data class representing a US Bank account to create a payment method for a recipient."""

    account_number: str
    routing_number: str
    recipient_account_id: str


@dataclass
class StripePayout:
    """Data class representing a Stripe payout."""

    from_account_id: str
    currency: str
    recipient_account_id: str
    destination_id: str
    amount: int


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


async def create_recipient_account(request: StripeRecipientCreateRequest) -> str:
    """Create a recipient account for a Stripe account.

    Args:
        request: The request to create a recipient account

    Returns:
        str: The ID of the created recipient account
    """
    try:
        log_dict = {"message": "Stripe: Creating recipient account", "request": json_dumps(request)}
        logging.info(json_dumps(log_dict))

        client = _get_stripe_client()

        legal_entity_data = {
            "business_type": "individual",
            "country": request.country.lower(),
            "representative": {"given_name": request.given_name, "surname": request.surname},
        }

        stripe_request = AccountService.CreateParams(
            include=["legal_entity_data", "configuration.recipient_data"],
            name=f"{request.given_name} {request.surname}",
            email=request.email,
            legal_entity_data=cast(AccountService.CreateParamsLegalEntityData, legal_entity_data),
            configuration={
                "recipient_data": {
                    "features": {
                        "bank_accounts": {"local": {"requested": True}, "wire": {"requested": True}},
                        "cards": {"requested": True},
                    }
                }
            },
        )

        response = client.v2.accounts.create(stripe_request)
        if not response:
            raise StripePayoutError("Failed to create recipient account", {"error": "No response from Stripe"})

        log_dict = {"message": "Stripe: Recipient account created", "response": json_dumps(response)}
        logging.info(json_dumps(log_dict))

        return response.id

    except Exception as e:
        log_dict = {"message": "Stripe: Error creating recipient account", "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise StripePayoutError("Failed to create recipient account", {"error": str(e)}) from e


async def create_stripe_us_bank_account(request: StripeUSBankAccountCreateRequest) -> str:
    """Create a Stripe US Bank account for a recipient.

    Args:
        request: The request to create a US Bank account

    Returns:
        str: The ID of the created US Bank account
    """
    try:
        log_dict = {"message": "Stripe: Creating Stripe US Bank account", "request": json_dumps(request)}
        logging.info(json_dumps(log_dict))

        client = _get_stripe_client()

        bank_account_data = cast(
            UsBankAccountService.CreateParams,
            {
                "account_number": request.account_number,
                "routing_number": request.routing_number,
            },
        )
        stripe_context = cast(RequestOptions, {"stripe_context": request.recipient_account_id})

        response = client.v2.payment_methods.us_bank_accounts.create(
            params=bank_account_data,
            options=stripe_context,
        )
        if not response:
            raise StripePayoutError("Failed to create Stripe US Bank account", {"error": "No response from Stripe"})

        log_dict = {"message": "Stripe: Stripe US Bank account created", "response": json_dumps(response)}
        logging.info(json_dumps(log_dict))

        return response.id

    except Exception as e:
        log_dict = {"message": "Stripe: Error creating Stripe US Bank account", "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise StripePayoutError("Failed to create Stripe US Bank account", {"error": str(e)}) from e


async def create_stripe_payout(request: StripePayout) -> tuple[str, str]:
    """Create a Stripe payout.

    Args:
        request: The request to create a Stripe payout

    Returns:
        tuple[str, str]: The ID of the created Stripe payout and the status of the created Stripe payout
    """
    try:
        log_dict = {"message": "Stripe: Creating Stripe payout", "request": json_dumps(request)}
        logging.info(json_dumps(log_dict))

        client = _get_stripe_client()

        amount_in_cents = int(request.amount * 100)  # Stripe requires amount in cents
        payout_params = cast(
            OutboundPaymentService.CreateParams,
            {
                "from": {
                    "financial_account": request.from_account_id,
                    "currency": request.currency.lower(),
                },
                "to": {
                    "recipient": request.recipient_account_id,
                    "destination": request.destination_id,
                    "currency": request.currency.lower(),
                },
                "amount": {
                    "currency": request.currency.lower(),
                    "value": amount_in_cents,
                },
            },
        )

        response = client.v2.outbound_payments.create(
            params=payout_params,
        )
        if not response:
            raise StripePayoutError("Failed to create Stripe payout", {"error": "No response from Stripe"})

        log_dict = {"message": "Stripe: Stripe payout created", "response": json_dumps(response)}
        logging.info(json_dumps(log_dict))

        return response.id, response.status

    except Exception as e:
        log_dict = {"message": "Stripe: Error creating Stripe payout", "error": str(e)}
        logging.error(json_dumps(log_dict))
        raise StripePayoutError("Failed to create Stripe payout", {"error": str(e)}) from e
