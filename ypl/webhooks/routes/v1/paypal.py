"""PayPal webhook handler module."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

import ypl.db.all_models  # noqa: F401
from fastapi import APIRouter, Header, Request
from paypalhttp import HttpError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload
from sqlmodel import select
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.payment.paypal.paypal_facilitator import PayPalFacilitator
from ypl.backend.payment.paypal.paypal_payout import TransactionStatus, _get_paypal_client
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import (
    PaymentTransaction,
    PaymentTransactionStatusEnum,
)
from ypl.db.webhooks import (
    WebhookDirectionEnum,
    WebhookEvent,
    WebhookPartner,
    WebhookPartnerStatusEnum,
    WebhookProcessingStatusEnum,
)

router = APIRouter(tags=["paypal"])

SLACK_WEBHOOK_CASHOUT = settings.SLACK_WEBHOOK_CASHOUT
WEBHOOK_ID = settings.PAYPAL_WEBHOOK_ID


class WebhookRequest:
    def __init__(self) -> None:
        self.path = "/v1/notifications/verify-webhook-signature"
        self.verb = "POST"
        self.headers = {"Content-Type": "application/json"}
        self.body: dict[str, Any] | None = None

    def request_body(self, body: dict[str, Any]) -> None:
        self.body = body


@router.post("/webhook/{token}")
async def handle_paypal_webhook(
    token: str,
    request: Request,
    paypal_auth_algo: str | None = Header(None, alias="PAYPAL-AUTH-ALGO"),
    paypal_cert_url: str | None = Header(None, alias="PAYPAL-CERT-URL"),
    paypal_transmission_sig: str | None = Header(None, alias="PAYPAL-TRANSMISSION-SIG"),
    paypal_transmission_id: str | None = Header(None, alias="PAYPAL-TRANSMISSION-ID"),
    paypal_transmission_time: str | None = Header(None, alias="PAYPAL-TRANSMISSION-TIME"),
) -> dict[str, Any]:
    """Handle incoming webhooks from PayPal.

    Args:
        token: The webhook token from the URL
        request: The incoming request
        paypal_auth_algo: The PAYPAL-AUTH-ALGO header for webhook verification
        paypal_cert_url: The PAYPAL-CERT-URL header for webhook verification
        paypal_transmission_sig: The PAYPAL-TRANSMISSION-SIG header for webhook verification
        paypal_transmission_id: The PAYPAL-TRANSMISSION-ID header for webhook verification
        paypal_transmission_time: The PAYPAL-TRANSMISSION-TIME header for webhook verification

    Returns:
        Dict[str, Any]: Response indicating success
    """
    # Return success immediately to prevent PayPal from retrying
    # Create background task to handle all processing
    try:
        payload = await request.body()
        log_dict = {
            "message": "PayPal webhook received",
            "token": token,
            "paypal_auth_algo": paypal_auth_algo,
            "paypal_cert_url": paypal_cert_url,
            "paypal_transmission_sig": paypal_transmission_sig,
            "paypal_transmission_id": paypal_transmission_id,
            "paypal_transmission_time": paypal_transmission_time,
            "payload": json_dumps(payload),
        }
        logging.info(json_dumps(log_dict))

        asyncio.create_task(
            process_paypal_webhook(
                token,
                payload,
                paypal_auth_algo,
                paypal_cert_url,
                paypal_transmission_sig,
                paypal_transmission_id,
                paypal_transmission_time,
            )
        )
    except Exception as err:
        # Log error but still return success to prevent retries
        log_dict = {
            "message": "PayPal Webhook: Error reading payload",
            "error": str(err),
        }
        logging.warning(json_dumps(log_dict))

    return {"status": "success"}


async def validate_token(token: str) -> bool:
    """Validate the webhook token against the webhook_partners table.

    Args:
        token: The webhook token from the URL

    Returns:
        bool: True if token is valid, False otherwise
    """
    if settings.ENVIRONMENT == "local":
        return True

    try:
        async with get_async_session() as session:
            stmt = select(WebhookPartner).where(
                WebhookPartner.webhook_token == token,
                WebhookPartner.status == WebhookPartnerStatusEnum.ACTIVE,
                WebhookPartner.name == "PAYPAL",
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
                return False

            return True

    except Exception as err:
        log_dict = {
            "message": "Error validating webhook token",
            "error": str(err),
            "token": token,
        }
        logging.error(json_dumps(log_dict))
        return False


async def validate_paypal_signature(
    payload: bytes,
    paypal_auth_algo: str | None,
    paypal_cert_url: str | None,
    paypal_transmission_sig: str | None,
    paypal_transmission_id: str | None,
    paypal_transmission_time: str | None,
) -> bool:
    """Validate the PayPal webhook signature and required headers.

    Args:
        payload: The raw request body
        paypal_auth_algo: The PAYPAL-AUTH-ALGO header
        paypal_cert_url: The PAYPAL-CERT-URL header
        paypal_transmission_sig: The PAYPAL-TRANSMISSION-SIG header
        paypal_transmission_id: The PAYPAL-TRANSMISSION-ID header
        paypal_transmission_time: The PAYPAL-TRANSMISSION-TIME header

    Returns:
        bool: True if signature is valid, False otherwise
    """
    if settings.ENVIRONMENT == "local":
        return True

    if not all(
        [paypal_transmission_sig, paypal_transmission_id, paypal_transmission_time, paypal_auth_algo, paypal_cert_url]
    ):
        log_dict = {
            "message": "PayPal Webhook: Missing required headers",
            "paypal_transmission_sig": paypal_transmission_sig,
            "paypal_transmission_id": paypal_transmission_id,
            "paypal_transmission_time": paypal_transmission_time,
            "paypal_auth_algo": paypal_auth_algo,
            "paypal_cert_url": paypal_cert_url,
        }
        logging.warning(json_dumps(log_dict))
        return False

    try:
        verification_request = {
            "transmission_id": paypal_transmission_id,
            "transmission_time": paypal_transmission_time,
            "cert_url": paypal_cert_url,
            "auth_algo": paypal_auth_algo,
            "transmission_sig": paypal_transmission_sig,
            "webhook_id": WEBHOOK_ID,
            "webhook_event": json.loads(payload),
        }

        client = _get_paypal_client()
        request = WebhookRequest()
        request.request_body(verification_request)

        response = client.execute(request)
        verification_status: str = response.result.verification_status

        log_dict = {
            "message": "PayPal Webhook: Signature verification result",
            "verification_status": verification_status,
            "paypal_transmission_id": paypal_transmission_id,
        }
        logging.info(json_dumps(log_dict))

        return verification_status == "SUCCESS"

    except HttpError as err:
        log_dict = {
            "message": "PayPal Webhook: HTTP error validating signature",
            "error": str(err),
            "status_code": getattr(err, "status_code", None),
        }
        logging.warning(json_dumps(log_dict))
        return False
    except Exception as err:
        log_dict = {
            "message": "PayPal Webhook: Error validating signature",
            "error": str(err),
        }
        logging.warning(json_dumps(log_dict))
        return False


async def upsert_webhook(
    webhook_event_id: str | None,
    event_data: dict[str, Any],
) -> WebhookEvent | None:
    """Check if a webhook event already exists or create a new one, using row-level locking.

    Args:
        webhook_event_id: The webhook event ID
        event_data: The webhook event data

    Returns:
        WebhookEvent | None: The webhook event or None if error
    """
    if not webhook_event_id:
        return None

    try:
        async with get_async_session() as session:
            paypal_partner_stmt = select(WebhookPartner).where(
                WebhookPartner.name == "PAYPAL",
                WebhookPartner.status == WebhookPartnerStatusEnum.ACTIVE,
                WebhookPartner.deleted_at.is_(None),  # type: ignore
            )
            result = await session.execute(paypal_partner_stmt)
            paypal_partner: WebhookPartner | None = result.scalar_one_or_none()

            if not paypal_partner:
                log_dict = {
                    "message": "PayPal Webhook: No active PayPal webhook partner found",
                    "webhook_event_id": webhook_event_id,
                }
                logging.warning(json_dumps(log_dict))
                return None

            stmt = (
                select(WebhookEvent)
                .where(
                    WebhookEvent.webhook_partner_id == paypal_partner.webhook_partner_id,
                    WebhookEvent.partner_webhook_reference_id == webhook_event_id,
                    WebhookEvent.deleted_at.is_(None),  # type: ignore
                )
                .with_for_update()
            )

            result = await session.execute(stmt)
            existing_event: WebhookEvent | None = result.scalar_one_or_none()

            if existing_event:
                log_dict = {
                    "message": "Duplicate PayPal webhook received",
                    "webhook_event_id": str(existing_event.webhook_event_id),
                    "processing_status": existing_event.processing_status.value,
                }
                logging.info(json_dumps(log_dict))
                return existing_event

            try:
                new_event = WebhookEvent(
                    webhook_partner_id=paypal_partner.webhook_partner_id,
                    direction=WebhookDirectionEnum.INCOMING,
                    raw_payload=event_data,
                    processing_status=WebhookProcessingStatusEnum.PENDING,
                    partner_webhook_reference_id=webhook_event_id,
                )
                session.add(new_event)
                await session.commit()

                log_dict = {
                    "message": "New PayPal webhook event created",
                    "webhook_event_id": str(new_event.webhook_event_id),
                    "processing_status": new_event.processing_status.value,
                }
                logging.info(json_dumps(log_dict))
                return new_event

            except IntegrityError:
                stmt = select(WebhookEvent).where(
                    WebhookEvent.webhook_partner_id == paypal_partner.webhook_partner_id,
                    WebhookEvent.partner_webhook_reference_id == webhook_event_id,
                    WebhookEvent.deleted_at.is_(None),  # type: ignore
                )
                result = await session.execute(stmt)
                duplicate_event: WebhookEvent | None = result.scalar_one_or_none()

                if duplicate_event:
                    log_dict = {
                        "message": "Found existing webhook event after IntegrityError",
                        "webhook_event_id": str(duplicate_event.webhook_event_id),
                        "processing_status": duplicate_event.processing_status.value,
                    }
                    logging.info(json_dumps(log_dict))
                    return duplicate_event
                return None

    except Exception as err:
        log_dict = {
            "message": "PayPal Webhook: Error checking/creating webhook",
            "error": str(err),
            "webhook_event_id": webhook_event_id,
        }
        logging.warning(json_dumps(log_dict))
        return None


async def update_webhook_event_status(webhook_event_id: UUID, status: WebhookProcessingStatusEnum) -> None:
    """Update the status of a webhook event.

    Args:
        webhook_event_id: The ID of the webhook event to update
        status: The new status to set
    """
    try:
        async with get_async_session() as session:
            stmt = select(WebhookEvent).where(
                WebhookEvent.webhook_event_id == webhook_event_id,
                WebhookEvent.deleted_at.is_(None),  # type: ignore
            )
            result = await session.execute(stmt)
            webhook_event: WebhookEvent | None = result.scalar_one_or_none()

            if webhook_event:
                webhook_event.processing_status = status
                webhook_event.modified_at = datetime.now()
                await session.commit()

                log_dict = {
                    "message": "PayPal Webhook: Webhook event status updated",
                    "webhook_event_id": str(webhook_event.webhook_event_id),
                    "status": status.value,
                }
                logging.info(json_dumps(log_dict))

    except Exception as err:
        log_dict = {
            "message": f"PayPal Webhook: Error updating webhook status to {status.value}",
            "error": str(err),
            "webhook_event_id": str(webhook_event_id),
        }
        logging.error(json_dumps(log_dict))
        asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))


async def process_paypal_webhook(
    token: str,
    payload: bytes,
    paypal_auth_algo: str | None,
    paypal_cert_url: str | None,
    paypal_transmission_sig: str | None,
    paypal_transmission_id: str | None,
    paypal_transmission_time: str | None,
) -> None:
    """Process the PayPal webhook asynchronously.

    Args:
        token: The webhook token from the URL
        payload: The raw request body
        paypal_auth_algo: The PAYPAL-AUTH-ALGO header for webhook verification
        paypal_cert_url: The PAYPAL-CERT-URL header for webhook verification
        paypal_transmission_sig: The PAYPAL-TRANSMISSION-SIG header for webhook verification
        paypal_transmission_id: The PAYPAL-TRANSMISSION-ID header for webhook verification
        paypal_transmission_time: The PAYPAL-TRANSMISSION-TIME header for webhook verification
    """
    is_valid_token = await validate_token(token)
    is_valid_signature = await validate_paypal_signature(
        payload,
        paypal_auth_algo,
        paypal_cert_url,
        paypal_transmission_sig,
        paypal_transmission_id,
        paypal_transmission_time,
    )

    event_data = json.loads(payload)

    if not is_valid_token or not is_valid_signature:
        log_dict = {
            "message": "PayPal Webhook: Validation failed",
            "is_valid_token": str(is_valid_token),
            "is_valid_signature": str(is_valid_signature),
            "token": token,
            "paypal_transmission_id": paypal_transmission_id,
            "event_data": event_data,
        }
        logging.warning(json_dumps(log_dict))
        asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
        return

    webhook_event = await upsert_webhook(event_data.get("id"), event_data)
    if webhook_event is None:
        log_dict = {
            "message": "PayPal Webhook: Error checking/creating webhook",
            "paypal_transmission_id": paypal_transmission_id,
            "event_data": event_data,
        }
        logging.error(json_dumps(log_dict))
        asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
        return

    if webhook_event.processing_status == WebhookProcessingStatusEnum.PROCESSED:
        log_dict = {
            "message": "PayPal Webhook: Skipping already processed webhook",
            "webhook_event_id": str(webhook_event.webhook_event_id),
            "paypal_transmission_id": paypal_transmission_id,
        }
        logging.info(json_dumps(log_dict))
        return

    await process_payout_event(event_data)
    await update_webhook_event_status(webhook_event.webhook_event_id, WebhookProcessingStatusEnum.PROCESSED)


async def process_payout_event(event_data: dict[str, Any]) -> None:
    """Process a PayPal payout event and update the corresponding transaction.

    Args:
        event_data: The webhook event data containing payout details
    """
    event_type = event_data.get("event_type")
    if not event_type or not event_type.startswith("PAYMENT.PAYOUTS-ITEM"):
        log_dict = {
            "message": "PayPal Webhook: Unsupported event type",
            "event_type": event_type,
            "event_data": event_data,
        }
        logging.warning(json_dumps(log_dict))
        return

    payment_transaction_id = event_data.get("resource", {}).get("payout_item", {}).get("sender_item_id")
    transaction_status = event_data.get("resource", {}).get("transaction_status")
    payout_batch_id = event_data.get("resource", {}).get("payout_batch_id")

    if not payment_transaction_id or not transaction_status or not payout_batch_id:
        log_dict = {
            "message": "PayPal Webhook: Missing required fields in event data",
            "event_type": event_type,
            "payment_transaction_id": payment_transaction_id,
            "transaction_status": transaction_status,
            "payout_batch_id": payout_batch_id,
            "event_data": event_data,
        }
        logging.warning(json_dumps(log_dict))
        return

    try:
        async with get_async_session() as session:
            stmt = (
                select(PaymentTransaction)
                .where(
                    PaymentTransaction.payment_transaction_id == payment_transaction_id,
                    PaymentTransaction.deleted_at.is_(None),  # type: ignore
                )
                .options(selectinload(PaymentTransaction.destination_instrument))  # type: ignore
            )
            result = await session.execute(stmt)
            transaction = result.scalar_one_or_none()

            if not transaction:
                log_dict = {
                    "message": "PayPal Webhook: No matching transaction found",
                    "payment_transaction_id": payment_transaction_id,
                    "payout_batch_id": payout_batch_id,
                    "transaction_status": transaction_status,
                }
                logging.warning(json_dumps(log_dict))
                return

            if transaction.status == PayPalFacilitator.map_paypal_status_to_internal(transaction_status):
                log_dict = {
                    "message": "PayPal Webhook: Transaction status already up to date",
                    "payment_transaction_id": payment_transaction_id,
                    "payout_batch_id": payout_batch_id,
                    "transaction_status": transaction_status,
                }
                logging.info(json_dumps(log_dict))
                return

            if transaction_status == TransactionStatus.SUCCESS:
                new_status = PaymentTransactionStatusEnum.SUCCESS
            elif transaction_status in [
                TransactionStatus.FAILED,
                TransactionStatus.REVERSED,
                TransactionStatus.CANCELED,
                TransactionStatus.DENIED,
                TransactionStatus.REFUNDED,
                TransactionStatus.RETURNED,
            ]:
                new_status = PaymentTransactionStatusEnum.FAILED
            else:
                new_status = PaymentTransactionStatusEnum.PENDING

            transaction.status = new_status
            transaction.modified_at = datetime.now()
            await session.commit()

            log_dict = {
                "message": "PayPal Webhook: Updated transaction status",
                "payment_transaction_id": str(transaction.payment_transaction_id),
                "payout_batch_id": payout_batch_id,
                "old_status": transaction.status.value,
                "new_status": new_status.value,
            }
            logging.info(json_dumps(log_dict))

            # Post to Slack for important status changes
            if new_status in [PaymentTransactionStatusEnum.SUCCESS, PaymentTransactionStatusEnum.FAILED]:
                asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

    except Exception as err:
        log_dict = {
            "message": "PayPal Webhook: Error processing payout event",
            "error": str(err),
            "payment_transaction_id": payment_transaction_id,
            "payout_batch_id": payout_batch_id,
            "transaction_status": transaction_status,
        }
        logging.error(json_dumps(log_dict))
        asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
