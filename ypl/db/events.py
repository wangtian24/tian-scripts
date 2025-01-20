from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel
from ypl.db.users import SYSTEM_USER_ID

if TYPE_CHECKING:
    from ypl.db.users import User


class Event(BaseModel, table=True):
    """Model for tracking system events."""

    __tablename__ = "events"

    event_id: UUID = Field(default_factory=uuid4, primary_key=True, nullable=False)

    user_id: str = Field(
        foreign_key="users.user_id",
        nullable=False,
        default=SYSTEM_USER_ID,
        index=True,
        description="The user who created this event",
    )

    event_name: str = Field(
        nullable=False,
        index=True,
        description="Name of the event",
    )

    event_category: str = Field(
        nullable=False,
        index=True,
        description="Category of the event for grouping",
    )

    event_params: dict | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="Additional parameters associated with the event",
    )

    event_guestivity_details: dict | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="Guestivity-specific details for the event",
    )

    event_dedup_id: str | None = Field(
        default=None,
        nullable=True,
        index=True,
        sa_type=sa.Text,
        description="Optional ID used for event deduplication",
    )

    user: "User" = Relationship(back_populates="events")
