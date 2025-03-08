import enum
import uuid
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel

if TYPE_CHECKING:
    from ypl.db.chats import ChatMessage
    from ypl.db.embeddings import MemoryEmbedding
    from ypl.db.users import User


# These are the various sources from which memory can be populated.
class MemorySource(enum.Enum):
    # A user's turn in the conversation.
    USER_MESSAGE = "user_message"
    # The assistant's turn in the conversation. Whether this is an
    # LLM or a Yapp is not specified here.
    ASSISTANT_MESSAGE = "assistant_message"


class ChatMessageMemoryAssociation(BaseModel, table=True):
    __tablename__ = "chat_message_memory_associations"

    association_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    memory_id: uuid.UUID = Field(foreign_key="memories.memory_id", primary_key=True, nullable=False)
    message_id: uuid.UUID = Field(foreign_key="chat_messages.message_id", primary_key=True, nullable=False)


class Memory(BaseModel, table=True):
    """
    Represents a 'memory' object. Each memory can store text, plus an embedding
    in a 1536-dimensional vector, generated by a particular embedding model.
    Optionally linked to the chat message that created it, a specific user,
    and an optional agent or 'yapp' that was used.
    """

    __tablename__ = "memories"

    # Needed for sa_type=TSVECTOR
    class Config:
        arbitrary_types_allowed = True

    memory_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)

    user_id: str = Field(foreign_key="users.user_id", nullable=False, index=True)

    # Content of the memory
    memory_content: str | None = Field(sa_column=Column(sa.Text, nullable=True), default=None)

    # The embedding vector.
    content_tsvector: TSVECTOR | None = Field(default=None, sa_column=Column(TSVECTOR))

    # The agent model that created this memory (if any).
    # Adjust the type to match your primary key field type for language_models.
    agent_language_model_id: uuid.UUID | None = Field(foreign_key="language_models.language_model_id", default=None)

    # The yapp that created this memory (if any).
    # Uncomment when we have a `yapps` table.
    # agent_yapp_id: uuid.UUID | None = Field(
    #     foreign_key="yapps.yapp_id",
    #     default=None
    # )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------
    user: "User" = Relationship(back_populates="memories")
    source_messages: list["ChatMessage"] = Relationship(
        back_populates="memories", link_model=ChatMessageMemoryAssociation
    )
    content_embedding: "MemoryEmbedding" = Relationship(
        back_populates="memory", sa_relationship_kwargs={"uselist": False}
    )
    # TODO(amin): implement the relationship below
    # agent_language_model: "LanguageModel" = Relationship()
