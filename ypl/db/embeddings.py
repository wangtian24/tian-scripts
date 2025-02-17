import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel
from ypl.db.chats import ChatMessage
from ypl.db.memories import Memory


class ChatMessageEmbedding(BaseModel, table=True):
    __tablename__ = "chat_message_embeddings"

    chat_message_embedding_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    message_id: uuid.UUID = Field(foreign_key="chat_messages.message_id", nullable=False, index=True)
    message: ChatMessage = Relationship(back_populates="content_embeddings")
    embedding: Vector | None = Field(sa_column=Column(Vector(1536), nullable=False))
    embedding_model_name: str = Field(nullable=False)

    class Config:
        arbitrary_types_allowed = True


class MemoryEmbedding(BaseModel, table=True):
    __tablename__ = "memory_embeddings"

    memory_embedding_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    memory_id: uuid.UUID = Field(foreign_key="memories.memory_id", nullable=False, index=True)
    memory: Memory = Relationship(back_populates="content_embedding")
    embedding: Vector | None = Field(sa_column=Column(Vector(1536), nullable=False))
    embedding_model_name: str = Field(nullable=False)

    class Config:
        arbitrary_types_allowed = True
