import uuid
from typing import Any

from fastapi import UploadFile
from pydantic import BaseModel as PydanticBaseModel
from sqlalchemy.dialects.postgresql import JSONB
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
    attachment_metadata: dict[str, Any] = Field(default_factory=dict, sa_type=JSONB, nullable=True)

    class Config:
        arbitrary_types_allowed = True


class TransientAttachment(PydanticBaseModel):
    filename: str
    content_type: str
    file: bytes

    class Config:
        arbitrary_types_allowed = True


async def convert_file_to_transient_file(file: UploadFile) -> TransientAttachment:
    if not file.filename or not file.content_type:
        raise ValueError("File has no filename or content type")
    return TransientAttachment(
        filename=file.filename,
        content_type=file.content_type,
        file=await file.read(),
    )
