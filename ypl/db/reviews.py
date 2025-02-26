import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Column
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship

from ypl.backend.llm.review_types import (
    BinaryResult,
    CritiqueResult,
    NuggetizedResult,
    ReviewStatus,
    ReviewType,
    SegmentedResult,
)
from ypl.db.base import BaseModel

if TYPE_CHECKING:
    from ypl.db.chats import ChatMessage


class MessageReview(BaseModel, table=True):
    __tablename__ = "message_reviews"
    review_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    message_id: uuid.UUID = Field(foreign_key="chat_messages.message_id", nullable=False, index=True)
    review_type: ReviewType = Field(sa_column=Column(SQLAlchemyEnum(ReviewType), nullable=False))
    result: BinaryResult | CritiqueResult | SegmentedResult | NuggetizedResult = Field(sa_type=JSONB)
    message: "ChatMessage" = Relationship(back_populates="reviews")
    reviewer_model_id: uuid.UUID = Field(foreign_key="language_models.language_model_id", nullable=False, index=True)
    status: ReviewStatus = Field(default=ReviewStatus.SUCCESS)
