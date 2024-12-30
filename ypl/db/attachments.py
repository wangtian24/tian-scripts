import uuid

from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel
from ypl.db.chats import ChatMessage


class Attachment(BaseModel, table=True):
    """
    Collection to store attachments for chat messages.
    Files are stored in google cloud storage and the URL and metadata is stored in the database.
    """

    __tablename__ = "attachments"
    attachment_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    chat_message_id: uuid.UUID | None = Field(foreign_key="chat_messages.message_id", index=True, nullable=True)
    chat_message: ChatMessage = Relationship(back_populates="attachments")
    file_name: str = Field(nullable=False)
    url: str = Field(nullable=False)
    content_type: str = Field(nullable=False)
    thumbnail_url: str | None = Field(default=None)

    class Config:
        arbitrary_types_allowed = True
