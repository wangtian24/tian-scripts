import enum
from uuid import UUID, uuid4

from sqlalchemy import Column
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel


class WebhookPartnerStatusEnum(enum.Enum):
    """Status of a webhook partner."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class WebhookDirectionEnum(enum.Enum):
    """Enum for webhook direction."""

    INCOMING = "incoming"
    OUTGOING = "outgoing"


class WebhookProcessingStatusEnum(enum.Enum):
    """Enum for webhook processing status."""

    INVALID = "invalid"
    PROCESSED = "processed"
    FAILED = "failed"
    PENDING = "pending"


class WebhookPartner(BaseModel, table=True):
    """Table for storing webhook partner information for incoming webhooks."""

    __tablename__ = "webhook_partners"

    webhook_partner_id: UUID = Field(default_factory=uuid4, primary_key=True, nullable=False)

    name: str = Field(
        nullable=False,
        index=True,
        description="Display name of the partner",
    )

    description: str | None = Field(
        default=None,
        description="Description of the partner's webhook integration",
    )

    webhook_token: str = Field(
        unique=True,
        index=True,
        nullable=False,
        description="URL-safe token used in webhook URL",
    )

    status: WebhookPartnerStatusEnum = Field(
        sa_column=Column(
            SQLAlchemyEnum(WebhookPartnerStatusEnum),
            nullable=False,
            default=WebhookPartnerStatusEnum.ACTIVE,
            server_default=WebhookPartnerStatusEnum.ACTIVE.name,
        ),
        description="Current status of the webhook integration",
    )

    validation_config: dict | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="Partner-specific validation rules (e.g., required fields, secret keys, etc.)",
    )

    webhook_events: list["WebhookEvent"] = Relationship(back_populates="webhook_partner")


class WebhookEvent(BaseModel, table=True):
    """Model for webhook events."""

    __tablename__ = "webhook_events"

    webhook_event_id: UUID = Field(default_factory=uuid4, primary_key=True, nullable=False)

    webhook_partner_id: UUID = Field(
        foreign_key="webhook_partners.webhook_partner_id",
        nullable=False,
        description="Reference to the webhook partner",
    )

    direction: WebhookDirectionEnum = Field(
        sa_column=Column(
            SQLAlchemyEnum(WebhookDirectionEnum),
            nullable=False,
            default=WebhookDirectionEnum.INCOMING,
            server_default=WebhookDirectionEnum.INCOMING.name,
        ),
        description="Direction of the webhook (incoming/outgoing)",
    )

    raw_payload: dict | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="Complete webhook payload",
    )

    processing_status: WebhookProcessingStatusEnum = Field(
        sa_column=Column(
            SQLAlchemyEnum(WebhookProcessingStatusEnum),
            nullable=False,
            default=WebhookProcessingStatusEnum.INVALID,
            server_default=WebhookProcessingStatusEnum.INVALID.name,
        ),
        description="Current processing status of the webhook",
    )

    partner_webhook_reference_id: str | None = Field(
        default=None,
        nullable=True,
        description="The reference ID from the partner's system for this webhook event",
    )

    webhook_partner: WebhookPartner = Relationship(back_populates="webhook_events")
