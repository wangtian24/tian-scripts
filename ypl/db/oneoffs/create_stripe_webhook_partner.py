"""Create a Stripe webhook partner entry.

This is a one-off script to create a webhook partner entry for Stripe
"""

from datetime import datetime
from secrets import token_urlsafe

from sqlalchemy import Connection
from sqlmodel import Session

from ypl.db.webhooks import WebhookPartner, WebhookPartnerStatusEnum


def create_stripe_webhook_partner(connection: Connection) -> None:
    """Create a webhook partner entry for Stripe if it doesn't exist."""
    with Session(connection) as session:
        existing_partner = (
            session.query(WebhookPartner)
            .filter(
                WebhookPartner.name == "STRIPE",
                WebhookPartner.deleted_at.is_(None),
                WebhookPartner.status == WebhookPartnerStatusEnum.ACTIVE,
            )
            .first()
        )

        if existing_partner:
            return

        webhook_token = token_urlsafe(32)
        partner = WebhookPartner(
            name="STRIPE",
            description="Stripe webhook integration for payout notifications",
            webhook_token=webhook_token,
            status=WebhookPartnerStatusEnum.ACTIVE,
            validation_config={
                "required_headers": ["Stripe-Signature"],
                "description": "Requires Stripe-Signature header for webhook validation",
            },
        )

        session.add(partner)
        session.commit()


def remove_stripe_webhook_partner(connection: Connection) -> None:
    """Soft delete the Stripe webhook partner entry by setting deleted_at."""
    with Session(connection) as session:
        partner = (
            session.query(WebhookPartner)
            .filter(WebhookPartner.name == "STRIPE", WebhookPartner.deleted_at.is_(None))
            .first()
        )

        if partner:
            partner.deleted_at = datetime.utcnow()
            session.commit()
