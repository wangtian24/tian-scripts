import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Integer, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import BaseModel

# The schema is based on the required authjs.dev prisma schema to minimize the custom logic required in
# the next js app.
# See https://authjs.dev/getting-started/database
# We will add our own adapter to handle login some day.


# Represents a user (human for now).
class User(BaseModel):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str | None] = mapped_column(Text)
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    # For magic link authentication, this is the timestamp of verification, or null if not using magic link.
    email_verified: Mapped[datetime | None] = mapped_column()
    image: Mapped[str | None] = mapped_column(Text)

    accounts: Mapped[list["Account"]] = relationship(back_populates="user")
    sessions: Mapped[list["Session"]] = relationship(back_populates="user")


# A user can have one or more accounts (for example, a user
# will have an account for Google OAuth and an account for Twitter).
class Account(BaseModel):
    __tablename__ = "accounts"

    provider: Mapped[str] = mapped_column(Text, primary_key=True)
    provider_account_id: Mapped[str] = mapped_column(Text, primary_key=True)

    user_id: Mapped[str] = mapped_column(
        Text, ForeignKey("users.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[str | None] = mapped_column(Text)
    access_token: Mapped[str | None] = mapped_column(Text)
    # Expiry of the access token, as unix timestamp in seconds.
    expires_at: Mapped[int | None] = mapped_column(Integer)
    token_type: Mapped[str | None] = mapped_column(Text)
    scope: Mapped[str | None] = mapped_column(Text)
    id_token: Mapped[str | None] = mapped_column(Text)
    session_state: Mapped[str | None] = mapped_column(Text)

    user: Mapped["User"] = relationship(back_populates="accounts")


class Session(BaseModel):
    __tablename__ = "sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_token: Mapped[str] = mapped_column(Text, unique=True)
    user_id: Mapped[str] = mapped_column(
        Text, ForeignKey("users.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False
    )
    expires: Mapped[datetime] = mapped_column(nullable=False)

    user: Mapped["User"] = relationship(back_populates="sessions")


# Verification tokens are used for pending magic link authentication.
class VerificationToken(BaseModel):
    __tablename__ = "verification_tokens"

    identifier: Mapped[str] = mapped_column(Text, primary_key=True)
    token: Mapped[str] = mapped_column(Text, primary_key=True)
    expires: Mapped[datetime] = mapped_column(nullable=False)
