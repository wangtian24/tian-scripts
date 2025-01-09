from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field

from ypl.db.base import BaseModel


class WebhookPartnerStatusEnum(str, Enum):
    """Status of a webhook partner."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class WebhookPartner(BaseModel, table=True):
    """Table for storing webhook partner information for incoming webhooks."""

    __tablename__ = "webhook_partners"

    webhook_partner_id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        nullable=False,
        description="Unique identifier for the webhook partner",
    )

    name: str = Field(
        nullable=False,
        index=True,
        description="Display name of the partner",
    )

    description: str | None = Field(
        default=None,
        description="Description of the partner's webhook integration",
    )

    # Webhook configuration
    webhook_token: str = Field(
        unique=True,
        index=True,
        nullable=False,
        description="URL-safe token used in webhook URL",
    )

    # Status
    status: WebhookPartnerStatusEnum = Field(
        sa_column=Column(
            SQLAlchemyEnum(WebhookPartnerStatusEnum),
            nullable=False,
            default=WebhookPartnerStatusEnum.ACTIVE,
            server_default=WebhookPartnerStatusEnum.ACTIVE.name,
        ),
        description="Current status of the webhook integration",
    )

    # Partner-specific validation rules
    validation_config: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB),
        description="Partner-specific validation rules (e.g., required fields, secret keys, etc.)",
    )

    class Config:
        arbitrary_types_allowed = True
