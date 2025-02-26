import enum
import uuid
from typing import Any

import sqlalchemy as sa
from sqlalchemy import Column, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel
from ypl.db.users import User


class EntityTypeEnum(enum.Enum):
    """Types of entities that can be added to the naughty list."""

    IP_ADDRESS = "IP_ADDRESS"
    DOMAIN = "DOMAIN"
    EMAIL = "EMAIL"
    USERNAME = "USERNAME"
    OTHER = "OTHER"


class StatusEnum(enum.Enum):
    """Status of a naughty list entry."""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class NaughtyList(BaseModel, table=True):
    """
    Represents entries in the naughty list - entities that have been identified as bad actors
    or have been found to abuse the system. This table is used to check against and prevent
    repeat attacks by stopping bad actors midway.
    """

    __tablename__ = "naughtylist"

    # Needed for JSONB type
    class Config:
        arbitrary_types_allowed = True

    # Primary key for the naughty list entry
    naughtylist_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        nullable=False,
        description="Unique identifier for the naughty list entry",
    )

    # The value of the entity (e.g., IP address, domain name, username, etc.)
    value: str = Field(
        nullable=False,
        sa_type=sa.Text,
        index=True,
        description="The value of the entity (e.g., IP address, domain name, username, etc.)",
    )

    # Type of entity
    entity_type: EntityTypeEnum = Field(
        sa_column=Column(
            sa.Enum(EntityTypeEnum),
            nullable=False,
        ),
        description="The type of entity being added to the naughty list",
    )

    # Reason for adding to the naughty list
    reason: str = Field(
        nullable=False, sa_type=sa.Text, description="The reason why this entity was added to the naughty list"
    )

    # Status of this entry
    status: StatusEnum = Field(
        default=StatusEnum.ACTIVE,
        sa_column=Column(
            sa.Enum(StatusEnum),
            nullable=False,
            server_default=StatusEnum.ACTIVE.value,
        ),
        description="Current status of the naughty list entry (ACTIVE or INACTIVE)",
    )

    # User who added this entry to the naughty list
    added_by_user_id: str | None = Field(
        foreign_key="users.user_id",
        nullable=True,
        description="ID of the user who added this entry to the naughty list",
    )

    # Additional metadata about this entity (JSON)
    entity_metadata: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=True),
        description="Additional JSON metadata about this entity",
    )

    # Relationships
    added_by: User | None = Relationship(
        sa_relationship_kwargs={"foreign_keys": "[NaughtyList.added_by_user_id]"},
    )

    # Ensure uniqueness of entity type + value combination
    __table_args__ = (UniqueConstraint("entity_type", "value", name="uq_naughtylist_entity_type_value"),)
