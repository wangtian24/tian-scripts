import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Any, Final
from uuid import UUID

from paypalhttp import HttpError
from paypalhttp.encoder import Encoder
from paypalhttp.serializers.json_serializer import Json
from paypalpayoutssdk.core import LiveEnvironment, PayPalHttpClient, SandboxEnvironment
from paypalpayoutssdk.payouts import PayoutsGetRequest, PayoutsPostRequest
from ypl.backend.config import settings
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import CurrencyEnum, PaymentInstrumentIdentifierTypeEnum

EMAIL_SUBJECT: Final[str] = "You have a credit from YUPP!"
EMAIL_MESSAGE: Final[str] = "You have received a credit from YUPP. Thanks for using YUPP!"


class TransactionStatus(StrEnum):
    """Enum for PayPal transaction statuses."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    DENIED = "DENIED"
    PENDING = "PENDING"
    UNCLAIMED = "UNCLAIMED"
    RETURNED = "RETURNED"
    ONHOLD = "ONHOLD"
    BLOCKED = "BLOCKED"
    REFUNDED = "REFUNDED"
    REVERSED = "REVERSED"
    UNKNOWN = "UNKNOWN"


GENERIC_ERROR_MESSAGE: Final[str] = "Internal error"


class PayPalPayoutError(Exception):
    """Custom exception for PayPal payout related errors."""

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
class PayPalPayout:
    """Represents a payout request for PayPal.

    Attributes:
        user_id: User ID
        amount: Amount to be paid out
        payment_transaction_id: UUID of the payment transaction
        currency: Type of currency
        destination_type: Type of destination (Paypal or Venmo)
        destination_identifier: Identifier of the destination (Paypal email or Venmo phone number)
    """

    amount: Decimal
    payment_transaction_id: UUID
    currency: CurrencyEnum
    destination_type: PaymentInstrumentIdentifierTypeEnum
    destination_identifier: str


def _get_paypal_client() -> PayPalHttpClient:
    """Get PayPal client instance."""

    paypal_config = settings.paypal_config

    if not all([paypal_config["client_id"], paypal_config["client_secret"]]):
        raise PayPalPayoutError(GENERIC_ERROR_MESSAGE, {"error": "Missing PayPal API credentials"})

    # Choose environment based on API URL
    if "sandbox" in paypal_config["api_url"]:
        environment = SandboxEnvironment(
            client_id=paypal_config["client_id"], client_secret=paypal_config["client_secret"]
        )
    else:
        environment = LiveEnvironment(
            client_id=paypal_config["client_id"], client_secret=paypal_config["client_secret"]
        )

    return PayPalHttpClient(environment)


async def process_paypal_payout(payout: PayPalPayout) -> tuple[str, str]:
    """Process a PayPal payout using the PayPal SDK.

    Args:
        payout: The payout to process

    Returns:
        Tuple[str, str]: A tuple containing (batch_id, batch_status)
    """
    log_dict: dict[str, Any] = {
        "message": "PayPal: Processing payout",
        "amount": str(payout.amount),
        "currency": str(payout.currency.value),
        "payment_transaction_id": str(payout.payment_transaction_id),
        "destination_type": str(payout.destination_type.value),
        "destination_identifier": str(payout.destination_identifier),
    }
    logging.info(json_dumps(log_dict))

    if not all(
        [
            payout.amount,
            payout.currency,
            payout.payment_transaction_id,
            payout.destination_type,
            payout.destination_identifier,
        ]
    ):
        validation_details: dict[str, Any] = {
            "message": "PayPal: Missing required fields",
            "amount": str(payout.amount),
            "currency": str(payout.currency.value),
            "payment_transaction_id": str(payout.payment_transaction_id),
            "destination_type": str(payout.destination_type.value),
            "destination_identifier": str(payout.destination_identifier),
            "error": "Missing required fields",
        }
        raise PayPalPayoutError(GENERIC_ERROR_MESSAGE, validation_details)

    # Round the amount to 2 decimal places for fiat currency
    rounded_amount = payout.amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    try:
        client = _get_paypal_client()

        if payout.destination_type == PaymentInstrumentIdentifierTypeEnum.PAYPAL_ID:
            recipient_wallet_type = "PAYPAL"
            recipient_type = "EMAIL"
        elif payout.destination_type == PaymentInstrumentIdentifierTypeEnum.VENMO_ID:
            recipient_wallet_type = "VENMO"
            recipient_type = "PHONE"

        body = {
            "sender_batch_header": {
                "sender_batch_id": str(payout.payment_transaction_id),
                "recipient_type": "EMAIL",
                "email_subject": EMAIL_SUBJECT,
                "email_message": EMAIL_MESSAGE,
            },
            "items": [
                {
                    "amount": {
                        "value": str(rounded_amount),
                        "currency": payout.currency.value.upper(),
                    },
                    "sender_item_id": str(payout.payment_transaction_id),
                    "recipient_wallet": recipient_wallet_type,
                    "receiver": payout.destination_identifier,
                    "recipient_type": recipient_type,
                    "purpose": "CASHBACK",
                }
            ],
        }

        log_dict = {
            "message": "PayPal: Calling PayPal API with Payout payload",
            "payload": json_dumps(body),
        }
        logging.info(json_dumps(log_dict))

        request = PayoutsPostRequest()
        request.request_body(body)

        response = client.execute(request)
        batch_id = response.result.batch_header.payout_batch_id
        batch_status = response.result.batch_header.batch_status

        if not batch_id:
            details = {
                "response_data": str(response.result),
                "payment_transaction_id": str(payout.payment_transaction_id),
                "error": "PayPal: Missing batch_id in payment response",
            }
            raise PayPalPayoutError(GENERIC_ERROR_MESSAGE, details)

        log_dict = {
            "message": "PayPal: Payout created",
            "amount": str(payout.amount),
            "currency": str(payout.currency.value),
            "payment_transaction_id": str(payout.payment_transaction_id),
            "destination_type": str(payout.destination_type.value),
            "destination_identifier": str(payout.destination_identifier),
            "batch_id": batch_id,
            "batch_status": batch_status,
        }
        logging.info(json_dumps(log_dict))

        return batch_id, batch_status

    except HttpError as e:
        encoder = Encoder([Json()])
        error = encoder.deserialize_response(e.message, e.headers)
        error_details = []
        for detail in error["details"]:
            error_detail = {
                "Error location": detail["location"],
                "Error field": detail["field"],
                "Error issue": detail["issue"],
            }
            error_details.append(error_detail)

        log_dict = {
            "message": "PayPal: HTTP error processing payout",
            "payment_transaction_id": str(payout.payment_transaction_id),
            "Error": error["name"],
            "Error message": error["message"],
            "Information link": error["information_link"],
            "Debug id": error["debug_id"],
            "Details": json_dumps(error_details),
        }
        logging.error(json_dumps(log_dict))
        raise PayPalPayoutError(str(e), {"details": json_dumps(error_details)}) from e
    except Exception as e:
        log_dict = {
            "message": "PayPal: Error processing payout",
            "payment_transaction_id": str(payout.payment_transaction_id),
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        details = {
            "message": "PayPal: Error processing payout",
            "payment_transaction_id": str(payout.payment_transaction_id),
            "error": str(e),
        }
        raise PayPalPayoutError(str(e), details) from e


async def get_transaction_status(batch_id: str) -> str:
    """Get the status of a specific payout batch.

    Args:
        batch_id: The ID of the payout batch to check

    Returns:
        str: The status of the payout transaction
    """
    try:
        client = _get_paypal_client()
        request = PayoutsGetRequest(batch_id)
        response = client.execute(request)
        batch_status = response.result.batch_header.batch_status
        items = response.result.items

        # Extract transaction status from the first item
        transaction_status = items[0].transaction_status if items else "UNKNOWN"

        log_dict = {
            "message": "PayPal: Retrieved batch status",
            "batch_id": batch_id,
            "batch_status": batch_status,
            "transaction_status": transaction_status,
            "items": json_dumps(items),
        }
        logging.info(json_dumps(log_dict))

        return str(transaction_status)

    except HttpError as e:
        details = {
            "batch_id": batch_id,
            "status_code": str(e.status_code),
            "error": str(e),
        }
        raise PayPalPayoutError("PayPal: Error getting payout status", details) from e
    except Exception as e:
        details = {"batch_id": batch_id, "error": str(e)}
        raise PayPalPayoutError("PayPal: Error getting payout status", details) from e
