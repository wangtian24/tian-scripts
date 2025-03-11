import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Column
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship

from ypl.backend.llm.review_types import (
    AllReviewResults,
    ReviewStatus,
    ReviewType,
)
from ypl.db.base import BaseModel

if TYPE_CHECKING:
    from ypl.db.chats import ChatMessage


class MessageReview(BaseModel, table=True):
    __tablename__ = "message_reviews"
    review_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    message_id: uuid.UUID = Field(foreign_key="chat_messages.message_id", nullable=False, index=True)
    review_type: ReviewType = Field(sa_column=Column(SQLAlchemyEnum(ReviewType), nullable=False))
    result: AllReviewResults = Field(sa_type=JSONB)
    message: "ChatMessage" = Relationship(back_populates="reviews")
    reviewer_model_id: uuid.UUID = Field(foreign_key="language_models.language_model_id", nullable=False, index=True)
    status: ReviewStatus = Field(default=ReviewStatus.SUCCESS)
    review_eval: "MessageReviewEval" = Relationship(
        back_populates="review", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class MessageReviewEval(BaseModel, table=True):
    __tablename__ = "message_review_evals"

    review_eval_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    review_id: uuid.UUID = Field(foreign_key="message_reviews.review_id", nullable=False, index=True)
    message_id: uuid.UUID = Field(foreign_key="chat_messages.message_id", nullable=False, index=True)
    review: MessageReview = Relationship(back_populates="review_eval")
    score: float = Field(nullable=True)
    user_comment: str | None = Field(nullable=True)
    message: "ChatMessage" = Relationship(back_populates="review_evals")
