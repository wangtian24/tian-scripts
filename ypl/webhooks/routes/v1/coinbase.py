"""Coinbase webhook handler module."""

import asyncio
import decimal
import hmac
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from typing import Any

import ypl.db.all_models  # noqa: F401
from fastapi import APIRouter, Header, Request
from sqlalchemy.orm import selectinload
from sqlmodel import select
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import PaymentTransaction, PaymentTransactionStatusEnum
from ypl.db.webhooks import WebhookPartner, WebhookPartnerStatusEnum

router = APIRouter(tags=["coinbase"])

SLACK_WEBHOOK_CASHOUT = settings.SLACK_WEBHOOK_CASHOUT


async def validate_token(token: str) -> None:
    """Validate the webhook token against the webhook_partners table.

    Args:
        token: The webhook token from the URL
    """
    if settings.ENVIRONMENT == "local":
        return

    try:
        async with get_async_session() as session:
            stmt = select(WebhookPartner).where(
                WebhookPartner.webhook_token == token,
                WebhookPartner.status == WebhookPartnerStatusEnum.ACTIVE,
                WebhookPartner.name == "COINBASE",
                WebhookPartner.deleted_at.is_(None),  # type: ignore
            )
            result = await session.exec(stmt)
            partner = result.one_or_none()

            if not partner:
                log_dict = {
                    "message": "Invalid or inactive webhook token",
                    "token": token,
                }
                logging.warning(json_dumps(log_dict))
                return

    except Exception as err:
        log_dict = {
            "message": "Error validating webhook token",
            "error": str(err),
            "token": token,
        }
        logging.error(json_dumps(log_dict))
        return


def validate_payload(payload: bytes, webhook_id: str, x_coinbase_signature: str) -> None:
    """Validate the Coinbase webhook payload using HMAC-SHA256.

    Args:
        payload: The raw request body
        webhook_id: The webhook ID from the payload
        x_coinbase_signature: The X-Coinbase-Signature header value
    """
    if settings.ENVIRONMENT == "local":
        return

    try:
        # Use webhook_id as the secret key
        webhook_secret = webhook_id.encode()

        # Use the raw payload as the message
        message = payload

        # Generate HMAC-SHA256 hash using webhook_id as key and payload as message
        calculated_signature = hmac.new(webhook_secret, message, sha256).hexdigest()

        if not hmac.compare_digest(calculated_signature, x_coinbase_signature):
            log_dict = {
                "message": "Coinbase Webhook: Invalid Coinbase webhook signature",
                "webhook_id": webhook_id,
                "x_coinbase_signature": x_coinbase_signature,
                "calculated_signature": calculated_signature,
            }
            logging.warning(json_dumps(log_dict))
            return

    except Exception as err:
        log_dict = {
            "message": "Coinbase Webhook: Error validating Coinbase webhook payload",
            "error": str(err),
            "webhook_id": webhook_id,
        }
        logging.warning(json_dumps(log_dict))
        return


def validate_signature(token: str, x_coinbase_signature: str | None) -> None:
    """Validate the Coinbase webhook signature header is present.

    Args:
        token: The webhook token from the URL
        x_coinbase_signature: The X-Coinbase-Signature header for webhook verification
    """
    if not x_coinbase_signature:
        log_dict = {
            "message": "Coinbase webhook signature is missing",
            "token": token,
        }
        logging.warning(json_dumps(log_dict))
        return


@router.post("/webhook/{token}")
async def handle_coinbase_webhook(
    token: str, request: Request, x_coinbase_signature: str | None = Header(None, alias="X-Coinbase-Signature")
) -> dict[str, Any]:
    """Handle incoming webhooks from Coinbase.

    Args:
        token: The webhook token from the URL
        request: The incoming request
        x_coinbase_signature: The X-Coinbase-Signature header for webhook verification

    Returns:
        Dict[str, Any]: Response indicating success
    """
    # Return success immediately to prevent Coinbase from retrying
    # Create background task to handle all processing
    try:
        payload = await request.body()
        asyncio.create_task(process_coinbase_webhook(token, payload, x_coinbase_signature))
    except Exception as err:
        # Log error but still return success to prevent retries
        log_dict = {
            "message": "Coinbase Webhook: Error reading payload",
            "error": str(err),
        }
        logging.warning(json_dumps(log_dict))

    return {"status": "success"}


async def process_coinbase_webhook(token: str, payload: bytes, x_coinbase_signature: str | None) -> None:
    """Process the Coinbase webhook asynchronously.

    Args:
        token: The webhook token from the URL
        payload: The raw request body
        x_coinbase_signature: The X-Coinbase-Signature header for webhook verification
    """
    validate_signature(token, x_coinbase_signature)

    try:
        event_data = json.loads(payload)
    except json.JSONDecodeError as err:
        log_dict = {
            "message": "Coinbase Webhook: Invalid JSON payload",
            "error": str(err),
        }
        logging.warning(json_dumps(log_dict))
        return

    webhook_id = event_data.get("webhookId")
    if not webhook_id:
        log_dict = {
            "message": "Coinbase Webhook: Missing webhookId",
            "event_data": event_data,
        }
        logging.warning(json_dumps(log_dict))
        return

    validate_payload(payload, webhook_id, x_coinbase_signature or "")
    await validate_token(token)

    log_dict = {
        "message": "Coinbase Webhook: Validated",
        "token": token,
        "event_data": event_data,
    }
    logging.info(json_dumps(log_dict))

    await potential_matching_pending_transactions(event_data)


async def potential_matching_pending_transactions(event_data: dict[str, Any]) -> None:
    """Find and log potential matching pending transactions for a given Coinbase webhook event.

    This function:
    1. First looks for exact transaction hash match in last 24 hours
    2. If no exact match, tries fuzzy matching based on address and value
    3. If no matches found, logs a warning and sends a critical alert

    For ERC20 transfer events, it specifically looks for:
    - Matching transaction hash (from transactionHash field)
    - Matching destination address (from to field)
    - Matching transfer amount (from value field)

    For transaction events, it specifically looks for:
    - Matching transaction hash (from transactionHash field)
    - Matching destination address (from to field)
    - Matching transfer amount (from value/valueString field)
    - Transaction status (from status field)

    Args:
        event_data: The Coinbase webhook event data containing transaction details
    """
    event_type = event_data.get("eventType")
    if event_type not in ["erc20_transfer", "transaction"]:
        log_dict = {
            "message": "Coinbase Webhook: Event type is not supported",
            "event_type": event_type,
            "event_data": event_data,
        }
        logging.warning(json_dumps(log_dict))
        return

    # Common fields for both event types
    transaction_hash = event_data.get("transactionHash")
    destination_address = event_data.get("to")
    value = event_data.get("value")

    required_fields = {
        "transaction_hash": transaction_hash,
        "destination_address": destination_address,
        "value": value,
    }

    # Additional fields for transaction event type
    if event_type == "transaction":
        status = event_data.get("status")
        required_fields["status"] = status

    if not all(required_fields.values()):
        log_dict = {
            "message": "Coinbase Webhook: Missing required fields in event data",
            "event_type": event_type,
            **required_fields,
            "event_data": event_data,
        }
        logging.warning(json_dumps(log_dict))
        return

    # Type assertions after validation
    assert isinstance(destination_address, str), "Destination address must be a string"
    assert isinstance(value, str), "Value must be a string"

    async with get_async_session() as session:
        # First, look for exact transaction hash match in last 24 hours
        exact_match_query = select(PaymentTransaction).where(
            PaymentTransaction.partner_reference_id == transaction_hash,
            PaymentTransaction.created_at >= datetime.now() - timedelta(hours=24),  # type: ignore
            PaymentTransaction.deleted_at.is_(None),  # type: ignore
        )
        result = await session.execute(exact_match_query)
        exact_match = result.scalar_one_or_none()

        if exact_match:
            log_dict = {
                "message": "Coinbase Webhook: Coinbase webhook found exact transaction hash match",
                "event_type": event_type,
                "webhook_transaction_hash": transaction_hash,
                "payment_transaction_id": str(exact_match.payment_transaction_id),
                "transaction_status": str(exact_match.status),
            }
            logging.info(json_dumps(log_dict))
            # log to slack if the transaction is pending
            log_dict[
                "message"
            ] = "Coinbase Webhook: :warning: Coinbase webhook found exact transaction hash match :warning:"
            if (
                exact_match.status == PaymentTransactionStatusEnum.PENDING
                or exact_match.status == PaymentTransactionStatusEnum.NOT_STARTED
            ):
                asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
            return

        # If no exact match, look for potential matches based on status, address, and value
        query = (
            select(PaymentTransaction)
            .options(selectinload(PaymentTransaction.destination_instrument))  # type: ignore
            .where(
                PaymentTransaction.status.not_in(  # type: ignore
                    [
                        PaymentTransactionStatusEnum.SUCCESS,
                        PaymentTransactionStatusEnum.REVERSED,
                        PaymentTransactionStatusEnum.FAILED,
                    ]
                ),
                PaymentTransaction.created_at >= datetime.now() - timedelta(hours=24),  # type: ignore
                PaymentTransaction.deleted_at.is_(None),  # type: ignore
            )
        )
        result = await session.execute(query)
        pending_transactions = list(result.scalars().all())

        potential_matches_found = False
        # Process each transaction with its already loaded payment instrument
        for txn in pending_transactions:
            payment_instrument = txn.destination_instrument
            if not payment_instrument or payment_instrument.deleted_at is not None:
                continue

            # Check for address match
            address_match = payment_instrument.identifier.lower() == destination_address.lower()

            # Check for value match
            value_match = False
            try:
                webhook_value = Decimal(value)
                txn_value = Decimal(str(txn.amount))

                if event_type == "erc20_transfer":
                    # For USDC: 1.2 USDC = 1200000 (6 decimals)
                    # Compare first 4 digits after normalization
                    normalized_webhook = webhook_value
                    normalized_txn = txn_value * Decimal("1e6")
                    value_match = str(normalized_webhook)[:4] == str(normalized_txn)[:4]
                else:
                    # For ETH: Need to handle wei to ETH conversion
                    # webhook_value is in wei, txn_value is in ETH (18 decimals)
                    # Compare first 4 digits after normalization
                    normalized_webhook = webhook_value
                    normalized_txn = txn_value * Decimal("1e18")
                    value_match = str(normalized_webhook)[:4] == str(normalized_txn)[:4]
            except (ValueError, TypeError, decimal.InvalidOperation):
                log_dict = {
                    "message": "Error converting values for comparison",
                    "webhook_value": value,
                    "txn_value": txn.amount,
                }
                logging.warning(json_dumps(log_dict))
                continue

            if address_match or value_match:
                potential_matches_found = True
                log_dict = {
                    "message": "Coinbase Webhook: :warning: Coinbase webhook found potential matching transaction",
                    "event_type": event_type,
                    "webhook_transaction_hash": transaction_hash,
                    "payment_transaction_id": str(txn.payment_transaction_id),
                    "address_match": address_match,
                    "value_match": value_match,
                    "webhook_address": destination_address,
                    "instrument_address": payment_instrument.identifier,
                    "webhook_value": value,
                    "transaction_amount": str(txn.amount),
                    "transaction_status": str(txn.status),
                }
                logging.info(json_dumps(log_dict))
                asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

        # If no potential matches found, log warning and send critical alert
        if not potential_matches_found:
            log_dict = {
                "message": "Coinbase Webhook: :x: NO MATCHING TRANSACTIONS FOUND :x:",
                "event_type": event_type,
                "webhook_transaction_hash": transaction_hash,
                "webhook_address": destination_address,
                "webhook_value": value,
                "search_window": "24 hours",
            }
            logging.warning(json_dumps(log_dict))
            asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
