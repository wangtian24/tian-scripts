import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlmodel import Field, Relationship

from db.base import BaseModel

# The schema is based on the required authjs.dev prisma schema to minimize the custom logic required in
# the next js app.
# See https://authjs.dev/getting-started/database
# We will add our own adapter to handle login some day.


# Represents a user (human for now).
class User(BaseModel, table=True):
    __tablename__ = "users"

    id: str = Field(primary_key=True, nullable=False, sa_type=sa.Text)
    name: str | None = Field(default=None, sa_type=sa.Text)
    email: str = Field(unique=True, nullable=False, sa_type=sa.Text)
    email_verified: datetime | None = Field(default=None)
    image: str | None = Field(default=None, sa_type=sa.Text)
    points: int = Field(default=0)

    accounts: list["Account"] = Relationship(back_populates="user", cascade_delete=True)
    sessions: list["Session"] = Relationship(back_populates="user", cascade_delete=True)


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
    user_id: str = Field(
        sa_column=sa.Column(sa.Text, sa.ForeignKey("users.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False)
    )
    expires: datetime = Field(nullable=False)

    user: User = Relationship(back_populates="sessions", cascade_delete=True)


class VerificationToken(BaseModel, table=True):
    __tablename__ = "verification_tokens"

    identifier: str = Field(primary_key=True, nullable=False, sa_type=sa.Text)
    token: str = Field(primary_key=True, nullable=False, sa_type=sa.Text)
    expires: datetime = Field(nullable=False)
