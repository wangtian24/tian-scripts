"""Create a Coinbase webhook partner entry.

This is a one-off script to create a webhook partner entry for Coinbase
with appropriate validation configuration for their webhook signature header.
"""

from datetime import datetime
from secrets import token_urlsafe

from sqlalchemy import Connection
from sqlmodel import Session

from ypl.db.webhooks import WebhookPartner, WebhookPartnerStatusEnum


def create_coinbase_webhook_partner(connection: Connection) -> None:
    """Create a webhook partner entry for Coinbase if it doesn't exist."""
    with Session(connection) as session:
        existing_partner = (
            session.query(WebhookPartner)
            .filter(WebhookPartner.name == "COINBASE", WebhookPartner.deleted_at.is_(None))
            .first()
        )

        if existing_partner:
            return

        webhook_token = token_urlsafe(32)
        partner = WebhookPartner(
            name="COINBASE",
            description="Coinbase webhook integration for payout notifications",
            webhook_token=webhook_token,
            status=WebhookPartnerStatusEnum.ACTIVE,
            validation_config={
                "required_headers": ["x-coinbase-signature"],
                "description": "Requires x-coinbase-signature header for webhook validation",
            },
        )

        session.add(partner)
        session.commit()


def remove_coinbase_webhook_partner(connection: Connection) -> None:
    """Soft delete the Coinbase webhook partner entry by setting deleted_at."""
    with Session(connection) as session:
        partner = (
            session.query(WebhookPartner)
            .filter(WebhookPartner.name == "COINBASE", WebhookPartner.deleted_at.is_(None))
            .first()
        )

        if partner:
            partner.deleted_at = datetime.utcnow()
            session.commit()
