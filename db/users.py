import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy import Column
from sqlmodel import Field, Relationship

from db.base import BaseModel
from db.point_transactions import PointTransaction

if TYPE_CHECKING:
    from db.chats import Chat, Eval, Turn


# The schema is based on the required authjs.dev prisma schema to minimize the custom logic required in
# the next js app.
# See https://authjs.dev/getting-started/database
# We will add our own adapter to handle login some day.


# Represents a user (both human and synthetic).
class User(BaseModel, table=True):
    __tablename__ = "users"

    id: str = Field(primary_key=True, nullable=False, sa_type=sa.Text)
    name: str | None = Field(default=None, sa_type=sa.Text)
    email: str = Field(unique=True, nullable=False, sa_type=sa.Text)
    email_verified: datetime | None = Field(default=None)
    image: str | None = Field(default=None, sa_type=sa.Text)
    points: int = Field(default=10000)

    backfill_job_id: uuid.UUID | None = Field(
        foreign_key="synthetic_backfill_attributes.id",
        default=None,
        nullable=True,
    )

    accounts: list["Account"] = Relationship(back_populates="user", cascade_delete=True)
    sessions: list["Session"] = Relationship(back_populates="user", cascade_delete=True)

    chats: list["Chat"] = Relationship(back_populates="creator")
    turns: list["Turn"] = Relationship(back_populates="creator")
    evals: list["Eval"] = Relationship(back_populates="user")

    synthetic_attributes: "SyntheticUserAttributes" = Relationship(back_populates="user", cascade_delete=True)
    backfill_attributes: "SyntheticBackfillAttributes" = Relationship(back_populates="generated_users")
    point_transactions: list[PointTransaction] = Relationship(back_populates="user", cascade_delete=True)


class SyntheticUserAttributes(BaseModel, table=True):
    """
    Represents attributes for synthetic users, such as the generating LLM, system prompt, temperature, etc. The presence
    of a user in this table indicates that the user is synthetic. For future optimization, it may make sense to
    denormalize this table and move the attributes to the User table to reduce the number of joins.
    """

    __tablename__ = "synthetic_user_attributes"

    user_id: str = Field(foreign_key="users.id", primary_key=True, nullable=False)
    persona: str = Field(nullable=False, sa_type=sa.Text, default="")
    interests: list[str] = Field(sa_column=Column(sa.ARRAY(sa.Text), nullable=False, default=[]))
    style: str = Field(nullable=False, sa_type=sa.Text, default="")

    user: "User" = Relationship(back_populates="synthetic_attributes")


class SyntheticBackfillAttributes(BaseModel, table=True):
    """Represents the attributes of a backfill job."""

    __tablename__ = "synthetic_backfill_attributes"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)

    num_users: int = Field(nullable=False, sa_type=sa.Integer)
    num_attempted_chats_per_user: int = Field(nullable=False, sa_type=sa.Integer)

    user_llm_model: str = Field(sa_type=sa.Text, nullable=False)
    user_llm_temperature: float = Field(nullable=False, sa_type=sa.Float)

    judge_models: list[str] = Field(sa_column=Column(sa.ARRAY(sa.Text), nullable=False, default=[]))
    judge_model_temperatures: list[float] = Field(sa_column=Column(sa.ARRAY(sa.Float), nullable=False, default=[]))
    git_commit_sha: str = Field(sa_type=sa.Text, nullable=False, default="")

    generated_users: list["User"] = Relationship(back_populates="backfill_attributes", cascade_delete=True)


class Account(BaseModel, table=True):
    __tablename__ = "accounts"

    provider: str = Field(primary_key=True, nullable=False, sa_type=sa.Text)
    provider_account_id: str = Field(primary_key=True, nullable=False, sa_type=sa.Text)

    user_id: str = Field(
        sa_column=sa.Column(sa.Text, sa.ForeignKey("users.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False)
    )
    type: str = Field(nullable=False, sa_type=sa.Text)
    refresh_token: str | None = Field(default=None, sa_type=sa.Text)
    access_token: str | None = Field(default=None, sa_type=sa.Text)
    expires_at: int | None = Field(default=None)
    token_type: str | None = Field(default=None, sa_type=sa.Text)
    scope: str | None = Field(default=None, sa_type=sa.Text)
    id_token: str | None = Field(default=None, sa_type=sa.Text)
    session_state: str | None = Field(default=None, sa_type=sa.Text)

    user: User = Relationship(back_populates="accounts")


class Session(BaseModel, table=True):
    __tablename__ = "sessions"

    session_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    session_token: str = Field(unique=True, sa_type=sa.Text)
    user_id: str = Field(sa_column=sa.Column(sa.Text, sa.ForeignKey("users.id"), nullable=False))
    expires: datetime = Field(nullable=False)

    user: User = Relationship(back_populates="sessions")


class VerificationToken(BaseModel, table=True):
    __tablename__ = "verification_tokens"

    identifier: str = Field(primary_key=True, nullable=False, sa_type=sa.Text)
    token: str = Field(primary_key=True, nullable=False, sa_type=sa.Text)
    expires: datetime = Field(nullable=False)
