"""Stripe webhook handler module."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, cast

from fastapi import APIRouter, Header, Request
from sqlmodel import select
from stripe import Event, SignatureVerificationError, Webhook
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack
from ypl.db.webhooks import (
    WebhookDirectionEnum,
    WebhookEvent,
    WebhookPartner,
    WebhookPartnerStatusEnum,
    WebhookProcessingStatusEnum,
)

router = APIRouter(tags=["stripe"])

SLACK_WEBHOOK_CASHOUT = settings.SLACK_WEBHOOK_CASHOUT


async def validate_stripe_signature(
    payload: bytes,
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
                    payload=payload.decode("utf-8"),
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
    log_dict = {
        "message": "Stripe webhook received",
        "webhook_token": webhook_token,
        "stripe_signature": stripe_signature,
    }
    logging.info(json.dumps(log_dict))
    try:
        payload = await request.body()
        log_dict = {
            "message": "Stripe webhook payload received",
            "webhook_token": webhook_token,
            "stripe_signature": stripe_signature,
        }
        logging.info(json.dumps(log_dict))
        asyncio.create_task(process_stripe_webhook(webhook_token, payload, stripe_signature))
    except Exception as err:
        # Log error but still return success to prevent retries
        log_dict = {
            "message": "Stripe Webhook: Error reading payload",
            "error": str(err),
        }
        logging.warning(json.dumps(log_dict))

    return {"status": "success"}


async def process_stripe_webhook(webhook_token: str, payload: bytes, stripe_signature: str | None) -> None:
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
        # Get the webhook partner
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

            # Validate signature and get event data
            event = await validate_stripe_signature(payload, stripe_signature, webhook_token)
            if not event:
                log_dict = {
                    "message": "Invalid Stripe signature or payload",
                    "webhook_token": webhook_token,
                    "stripe_signature": stripe_signature,
                }
                logging.error(json.dumps(log_dict))
                return

            event_type = event.type
            event_id = event.id
            event_data = event.data.object

            # Create webhook event record
            webhook_event = WebhookEvent(
                webhook_partner_id=partner.webhook_partner_id,
                direction=WebhookDirectionEnum.INCOMING,
                raw_payload=event_data,
                processing_status=WebhookProcessingStatusEnum.PENDING,
                partner_webhook_reference_id=event_id,
            )
            session.add(webhook_event)
            await session.commit()

            # Process specific event types
            log_dict = {
                "message": "Stripe webhook event created",
                "webhook_event_id": str(webhook_event.webhook_event_id),
                "event_id": event_id,
                "event_type": event_type,
                "event_data": json.dumps(event_data),
            }
            logging.info(json.dumps(log_dict))
            # Update event status
            webhook_event.processing_status = WebhookProcessingStatusEnum.PROCESSED
            webhook_event.processed_at = datetime.utcnow()
            await session.commit()

            # TODO: Post notification to Slack for important events

    except Exception as e:
        log_dict = {
            "message": "Error processing Stripe webhook",
            "error": str(e),
            "webhook_token": webhook_token,
        }
        logging.warning(json.dumps(log_dict))
        asyncio.create_task(post_to_slack(json.dumps(log_dict), SLACK_WEBHOOK_CASHOUT))
