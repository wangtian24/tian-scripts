"""Stripe webhook handler module."""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, cast

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from stripe import Event, SignatureVerificationError, Webhook
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack_bg
from ypl.backend.payment.payout_utils import handle_failed_transaction
from ypl.backend.utils.async_utils import create_background_task
from ypl.db.payments import PaymentTransaction, PaymentTransactionStatusEnum
from ypl.db.point_transactions import PointTransaction
from ypl.db.webhooks import (
    WebhookDirectionEnum,
    WebhookEvent,
    WebhookPartner,
    WebhookPartnerStatusEnum,
    WebhookProcessingStatusEnum,
)


class StripeOutboundPaymentEventEnum(str, Enum):
    """Enum for Stripe outbound payment event types."""

    CANCELED = "outbound_payment.canceled"
    CREATED = "outbound_payment.created"
    FAILED = "outbound_payment.failed"
    POSTED = "outbound_payment.posted"
    RETURNED = "outbound_payment.returned"


router = APIRouter(tags=["stripe"])

SLACK_WEBHOOK_CASHOUT = settings.SLACK_WEBHOOK_CASHOUT


async def validate_stripe_ip(request: Request) -> None:
    """Validate that the request is coming from an allowed Stripe IP.

    Args:
        request: FastAPI request object

    Raises:
        HTTPException: If the request IP is not in the allowed list
    """
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else None

    if not client_ip and request.client:
        client_ip = request.client.host

    if not client_ip:
        logging.error("Could not determine client IP address")
        raise HTTPException(status_code=400, detail="Could not determine client IP")

    stripe_config = settings.STRIPE_CONFIG
    allowed_ips = stripe_config.get("webhook_allowed_ips", [])

    if not allowed_ips:
        logging.warning("No Stripe webhook allowed IPs configured")
        return

    if client_ip not in allowed_ips:
        log_dict = {
            "message": "Stripe webhook: Unauthorized IP for Stripe webhook",
            "ip": client_ip,
            "allowed_ips": allowed_ips,
        }
        logging.warning(json.dumps(log_dict))
        raise HTTPException(status_code=403, detail="Unauthorized IP address")


async def validate_stripe_signature(
    payload: str,
    signature: str,
    webhook_token: str,
) -> Event | None:
    """Validate the Stripe webhook signature.

    Args:
        payload: Raw request payload
        signature: Stripe signature from headers
        webhook_token: Webhook token for the partner

    Returns:
        Event | None: The validated Stripe event object or None if validation fails
    """
    try:
        webhook_secret = settings.STRIPE_CONFIG.get("webhook_secret")
        if not webhook_secret:
            logging.error("Stripe webhook secret not configured")
            return None

        try:
            event = cast(
                Event,
                Webhook.construct_event(
                    payload=payload,
                    sig_header=signature,
                    secret=webhook_secret,
                ),
            )
            return event

        except (ValueError, SignatureVerificationError) as e:
            logging.error(f"Invalid Stripe webhook validation: {str(e)}")
            return None

    except Exception as e:
        logging.error(f"Unexpected error in Stripe signature validation: {str(e)}")
        return None


@router.post("/webhook/{webhook_token}")
async def handle_stripe_webhook(
    request: Request,
    webhook_token: str,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
    _: None = Depends(validate_stripe_ip),
) -> dict[str, Any]:
    """Handle incoming Stripe webhook events.

    Args:
        request: FastAPI request object
        webhook_token: Token identifying the webhook partner
        stripe_signature: Stripe signature header

    Returns:
        dict: Response indicating webhook processing status
    """
    # Return success immediately to prevent retries
    # Create background task to handle all processing
    try:
        payload = (await request.body()).decode("utf-8")
        log_dict = {
            "message": "Stripe webhook payload received",
            "webhook_token": webhook_token,
            "stripe_signature": stripe_signature,
            "payload": json.dumps(payload),
            "for_id": json.loads(payload)["related_object"]["id"],
        }
        logging.info(json.dumps(log_dict))
        create_background_task(process_stripe_webhook(webhook_token, payload, stripe_signature))
    except Exception as err:
        # Log error but still return success to prevent retries
        log_dict = {
            "message": "Stripe Webhook: Error reading payload",
            "error": str(err),
        }
        logging.warning(json.dumps(log_dict))

    return {"status": "success"}


async def process_stripe_webhook(webhook_token: str, payload: str, stripe_signature: str | None) -> None:
    """Process the Stripe webhook asynchronously.

    Args:
        webhook_token: Token identifying the webhook partner
        payload: Raw request payload
        stripe_signature: Stripe signature header
    """
    if not stripe_signature:
        log_dict = {
            "message": "Stripe webhook signature is missing",
            "webhook_token": webhook_token,
        }
        logging.warning(json.dumps(log_dict))
        return

    try:
        async with get_async_session() as session:
            partner = (
                await session.execute(
                    select(WebhookPartner).where(
                        WebhookPartner.webhook_token == webhook_token,
                        WebhookPartner.deleted_at.is_(None),  # type: ignore
                        WebhookPartner.status == WebhookPartnerStatusEnum.ACTIVE,
                    )
                )
            ).scalar_one_or_none()

            if not partner:
                log_dict = {
                    "message": "Invalid webhook token",
                    "webhook_token": webhook_token,
                }
                logging.error(json.dumps(log_dict))
                return

            # Validate signature and ensure valid event data
            event = await validate_stripe_signature(payload, stripe_signature, webhook_token)
            if not event:
                log_dict = {
                    "message": "Invalid Stripe signature or payload",
                    "webhook_token": webhook_token,
                    "stripe_signature": stripe_signature,
                }
                logging.error(json.dumps(log_dict))
                return

            payload_data = json.loads(payload)
            event_type = payload_data["type"]
            related_object_id = payload_data["related_object"]["id"]

            # Create webhook event record
            webhook_event = WebhookEvent(
                webhook_partner_id=partner.webhook_partner_id,
                direction=WebhookDirectionEnum.INCOMING,
                raw_payload=payload_data,
                processing_status=WebhookProcessingStatusEnum.PENDING,
                partner_webhook_reference_id=event.id,
            )
            session.add(webhook_event)
            await session.commit()

            # Process specific event types
            log_dict = {
                "message": "Stripe webhook event created",
                "webhook_event_id": str(webhook_event.webhook_event_id),
                "event_id": event.id,
                "event_type": event_type,
                "event_data": json.dumps(payload_data),
                "for_id": related_object_id,
            }
            logging.info(json.dumps(log_dict))

            #  call a method to update the payment information based on the event type
            await update_payment_information(event_type, related_object_id)

            # Update event status
            webhook_event.processing_status = WebhookProcessingStatusEnum.PROCESSED
            webhook_event.modified_at = datetime.now()
            await session.commit()

            # TODO: Post notification to Slack for important events

    except IntegrityError:
        # If we hit a unique violation, another process created the record
        pass
    except Exception as e:
        log_dict = {
            "message": "Error processing Stripe webhook",
            "error": str(e),
            "webhook_token": webhook_token,
        }
        logging.warning(json.dumps(log_dict))
        post_to_slack_bg(json.dumps(log_dict), SLACK_WEBHOOK_CASHOUT)


async def update_payment_information(event_type: StripeOutboundPaymentEventEnum, related_object_id: str) -> None:
    """Update payment information based on the event type.

    Args:
        event_type: Stripe event type
        related_object_id: ID of the related object
    """

    #  retrieve the payment transaction id for this related object id
    async with get_async_session() as session:
        query_result = await session.execute(
            select(PaymentTransaction).where(PaymentTransaction.partner_reference_id == related_object_id)
        )
        payment_transaction = query_result.scalar_one_or_none()

        if not payment_transaction:
            log_dict = {
                "message": "Stripe webhook: Payment transaction not found in our db for an incoming event",
                "related_object_id": related_object_id,
            }
            logging.error(json.dumps(log_dict))
            post_to_slack_bg(json.dumps(log_dict), SLACK_WEBHOOK_CASHOUT)
            return

    if event_type == StripeOutboundPaymentEventEnum.POSTED:
        if payment_transaction.status == PaymentTransactionStatusEnum.SUCCESS:
            log_dict = {
                "message": "Stripe webhook: Payment transaction already marked success",
                "related_object_id": related_object_id,
            }
            logging.info(json.dumps(log_dict))
            return

        if payment_transaction.status == PaymentTransactionStatusEnum.PENDING:
            log_dict = {
                "message": "Stripe webhook: Payment transaction status is pending and setting to success",
                "related_object_id": related_object_id,
            }
            logging.info(json.dumps(log_dict))
            payment_transaction.status = PaymentTransactionStatusEnum.SUCCESS
            payment_transaction.modified_at = datetime.now()
            await session.commit()
        else:
            log_dict = {
                "message": (
                    "Stripe webhook: Payment transaction status is posted but not pending or success. "
                    "Check the status if it was reversed or failed before and is now getting marked as success"
                ),
                "related_object_id": related_object_id,
                "payment_transaction_status": payment_transaction.status,
                "payment_transaction_id": payment_transaction.payment_transaction_id,
                "webhook_status": event_type,
            }
            logging.error(json.dumps(log_dict))
            post_to_slack_bg(json.dumps(log_dict), SLACK_WEBHOOK_CASHOUT)
    elif event_type in [
        StripeOutboundPaymentEventEnum.CANCELED,
        StripeOutboundPaymentEventEnum.FAILED,
        StripeOutboundPaymentEventEnum.RETURNED,
    ]:
        log_dict = {
            "message": "Stripe webhook: Payment transaction failed",
            "payment status": event_type,
            "related_object_id": related_object_id,
            "payment_transaction_status": payment_transaction.status,
            "payment_transaction_id": payment_transaction.payment_transaction_id,
            "webhook_status": event_type,
        }
        logging.info(json.dumps(log_dict))

        if payment_transaction.status == PaymentTransactionStatusEnum.REVERSED:
            log_dict = {
                "message": "Stripe webhook: Payment transaction already reversed",
                "related_object_id": related_object_id,
            }
            logging.info(json.dumps(log_dict))
            return

        post_to_slack_bg(json.dumps(log_dict), SLACK_WEBHOOK_CASHOUT)
        # retrieve the other data required to handle failed transaction
        async with get_async_session() as session:
            query_result = await session.execute(
                select(PointTransaction).where(
                    PointTransaction.cashout_payment_transaction_id == payment_transaction.payment_transaction_id
                )
            )
            point_transaction = query_result.scalar_one_or_none()

            if not point_transaction:
                log_dict = {
                    "message": "Stripe webhook: Point transaction not found",
                    "related_object_id": related_object_id,
                }
                logging.error(json.dumps(log_dict))
                post_to_slack_bg(json.dumps(log_dict), SLACK_WEBHOOK_CASHOUT)
                return

        await handle_failed_transaction(
            payment_transaction_id=payment_transaction.payment_transaction_id,
            points_transaction_id=point_transaction.transaction_id,
            user_id=point_transaction.user_id,
            credits_to_cashout=point_transaction.point_delta,
            amount=payment_transaction.amount,
            usd_amount=payment_transaction.usd_amount,
            source_instrument_id=payment_transaction.source_instrument_id,
            destination_instrument_id=payment_transaction.destination_instrument_id,
            destination_identifier=payment_transaction.destination_identifier,
            destination_identifier_type=payment_transaction.destination_identifier_type,
            update_points=True,
            currency=payment_transaction.currency,
        )
    elif event_type == StripeOutboundPaymentEventEnum.CREATED:
        log_dict = {
            "message": "Stripe webhook: Payment transaction created",
            "related_object_id": related_object_id,
        }
        logging.info(json.dumps(log_dict))
    else:
        log_dict = {
            "message": "Stripe webhook: Unknown event type",
            "event_type": event_type,
        }
        logging.error(json.dumps(log_dict))
        post_to_slack_bg(json.dumps(log_dict), SLACK_WEBHOOK_CASHOUT)
