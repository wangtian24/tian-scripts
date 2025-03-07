import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Column
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel
from ypl.db.users import User


class AbuseActionType(enum.Enum):
    # The actions to take when an abuse event is detected.
    SLACK_REPORT = "slack_report"
    RAISE_EXCEPTION = "raise_exception"
    DEACTIVATE_USER = "deactivate_user"


class AbuseEventState(enum.Enum):
    # The state of an abuse event.
    PENDING_REVIEW = "pending_review"
    REVIEWED = "reviewed"
    CANCELLED = "cancelled"


class AbuseEventType(enum.Enum):
    # The type of abuse reported by an event.
    CASHOUT_SAME_INSTRUMENT_AS_REFERRER = "cashout_same_instrument_as_referrer"
    CASHOUT_SAME_INSTRUMENT_AS_RECENT_NEW_USER = "cashout_same_instrument_as_recent_new_user"
    CASHOUT_MULTIPLE_RECENT_REFERRAL_SIGNUPS = "cashout_multiple_recent_referral_signups"
    SIGNUP_SAME_EMAIL_AS_EXISTING_USER = "signup_same_email_as_existing_user"
    SIGNUP_SIMILAR_EMAIL_AS_REFERRER = "signup_similar_eamil_as_referrer"
    SIGNUP_SIMILAR_EMAIL_AS_RECENT_USER = "signup_similar_email_as_recent_user"
    SIGNUP_SIMILAR_NAME_AS_REFERRER = "signup_similar_name_as_referrer"
    SIGNUP_SIMILAR_NAME_AS_RECENT_USER = "signup_similar_name_as_recent_user"
    ACTIVITY_VOLUME = "activity_volume"
    CONTENT_LOW_QUALITY_MODEL_FEEDBACK = "content_low_quality_model_feedback"


class AbuseEvent(BaseModel, table=True):
    __tablename__ = "abuse_events"

    abuse_event_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    user_id: str = Field(foreign_key="users.user_id", nullable=False, index=True)
    user: User = Relationship(back_populates="abuse_events")

    event_type: AbuseEventType = Field(nullable=False)
    event_details: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))

    state: AbuseEventState = Field(
        sa_column=Column(
            SQLAlchemyEnum(AbuseEventState),
            nullable=False,
            default=AbuseEventState.PENDING_REVIEW,
            server_default=AbuseEventState.PENDING_REVIEW.name,
        )
    )

    reviewed_at: datetime | None = Field(nullable=True)
    reviewed_by: str | None = Field(nullable=True)
    review_notes: str | None = Field(nullable=True)
