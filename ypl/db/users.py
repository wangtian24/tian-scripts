import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy import Column, UniqueConstraint
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel
from ypl.db.point_transactions import PointTransaction
from ypl.db.rewards import Reward, RewardActionLog

if TYPE_CHECKING:
    from ypl.db.chats import Chat, Eval, Turn
    from ypl.db.language_models import LanguageModel

# The time threshold for considering a user as "new" from user created_at
# Users signed up within this time are treated as new users
NEW_USER_THRESHOLD = timedelta(days=3)

# The time threshold for considering a user as "inactive"
# Users who haven't had activity longer than this are considered inactive
INACTIVE_USER_THRESHOLD = timedelta(weeks=2)

# The default user is SYSTEM.
SYSTEM_USER_ID = "SYSTEM"

# The schema is based on the required authjs.dev prisma schema to minimize the custom logic required in
# the next js app.
# See https://authjs.dev/getting-started/database
# We will add our own adapter to handle login some day.


# Represents a user (both human and synthetic).
class User(BaseModel, table=True):
    __tablename__ = "users"

    user_id: str = Field(primary_key=True, nullable=False, sa_type=sa.Text)
    name: str | None = Field(default=None, sa_type=sa.Text)

    # Forcing the pre-convention constraint name for backwards compatibility.
    email: str = Field(sa_column=Column("email", sa.Text, nullable=False))
    __table_args__ = (UniqueConstraint("email", name="users_email_key"),)

    email_verified: datetime | None = Field(default=None)
    image: str | None = Field(default=None, sa_type=sa.Text)
    points: int = Field(default=10000, sa_column=Column(sa.Integer, server_default="10000", nullable=False))

    backfill_job_id: uuid.UUID | None = Field(
        foreign_key="synthetic_backfill_attributes.id",
        default=None,
        nullable=True,
    )

    # user who created this user
    creator_user_id: str | None = Field(foreign_key="users.user_id", default=None, nullable=True)
    user_creator: "User" = Relationship(
        back_populates="created_users", sa_relationship_kwargs={"remote_side": "User.user_id"}
    )
    created_users: list["User"] = Relationship(back_populates="user_creator")
    created_language_models: list["LanguageModel"] = Relationship(back_populates="language_model_creator")

    accounts: list["Account"] = Relationship(back_populates="user", cascade_delete=True)
    sessions: list["Session"] = Relationship(back_populates="user", cascade_delete=True)

    chats: list["Chat"] = Relationship(back_populates="creator")
    turns: list["Turn"] = Relationship(back_populates="creator")
    evals: list["Eval"] = Relationship(back_populates="user")

    synthetic_attributes: "SyntheticUserAttributes" = Relationship(back_populates="user", cascade_delete=True)
    backfill_attributes: "SyntheticBackfillAttributes" = Relationship(back_populates="generated_users")
    point_transactions: list[PointTransaction] = Relationship(back_populates="user", cascade_delete=True)
    reward_action_logs: list["RewardActionLog"] = Relationship(back_populates="user", cascade_delete=True)
    rewards: list["Reward"] = Relationship(back_populates="user", cascade_delete=True)

    def is_new_user(self) -> bool:
        return (datetime.now(UTC) - self.created_at) <= NEW_USER_THRESHOLD  # type: ignore

    def is_inactive_user(self) -> bool:
        latest_activity_at = self.get_latest_activity()
        if latest_activity_at is None:
            return True
        return (datetime.now(UTC) - latest_activity_at) > INACTIVE_USER_THRESHOLD

    def get_latest_activity(self) -> datetime | None:
        if not self.turns:
            return self.created_at
        return max((turn.created_at for turn in self.turns if turn.created_at is not None), default=self.created_at)


class SyntheticUserAttributes(BaseModel, table=True):
    """
    Represents attributes for synthetic users, such as the generating LLM, system prompt, temperature, etc. The presence
    of a user in this table indicates that the user is synthetic. For future optimization, it may make sense to
    denormalize this table and move the attributes to the User table to reduce the number of joins.
    """

    __tablename__ = "synthetic_user_attributes"

    user_id: str = Field(foreign_key="users.user_id", primary_key=True, nullable=False)
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
        sa_column=sa.Column(
            sa.Text, sa.ForeignKey("users.user_id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False
        )
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

    # Forcing the pre-convention constraint name for backwards compatibility.
    session_token: str = Field(sa_column=Column("session_token", sa.Text, nullable=False))
    __table_args__ = (UniqueConstraint("session_token", name="sessions_session_token_key"),)

    user_id: str = Field(sa_column=sa.Column(sa.Text, sa.ForeignKey("users.user_id"), nullable=False))
    expires: datetime = Field(nullable=False)

    user: User = Relationship(back_populates="sessions")


class VerificationToken(BaseModel, table=True):
    __tablename__ = "verification_tokens"

    identifier: str = Field(primary_key=True, nullable=False, sa_type=sa.Text)
    token: str = Field(primary_key=True, nullable=False, sa_type=sa.Text)
    expires: datetime = Field(nullable=False)
