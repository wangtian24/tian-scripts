import uuid

import sqlalchemy as sa
from sqlmodel import Field

from ypl.db.base import BaseModel


class AppFeedback(BaseModel, table=True):
    __tablename__ = "app_feedback"

    feedback_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="users.user_id", nullable=False, sa_type=sa.Text, index=True)
    chat_id: uuid.UUID | None = Field(foreign_key="chats.chat_id", nullable=True, index=True)
    user_comment: str | None = Field(nullable=False)
