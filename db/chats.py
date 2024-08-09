import enum
import uuid

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import BaseModel


# A chat can contain multiple conversations.
class Chat(BaseModel):
    __tablename__ = "chats"

    chat_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    # URL segment for the chat.
    # This field is world visible, do not leak information in this field.
    path = mapped_column(Text, nullable=False, unique=True, index=True)
    turns: Mapped[list["Turn"]] = relationship(
        back_populates="chat",
        order_by="Turn.sequence",
    )
    # TODO(minqi): define user relationship


class Turn(BaseModel):
    __tablename__ = "turns"

    turn_id = mapped_column(Uuid(as_uuid=True), primary_key=True, nullable=False, default=uuid.uuid4)

    chat_id = mapped_column(Uuid(as_uuid=True), ForeignKey("chats.chat_id"), nullable=False)
    chat = relationship("Chat", back_populates="turns")

    # Sequence of the turn in the chat. This sequence is not guaranteed to be
    # continuous, as messages may be deleted in the future.
    sequence_id = mapped_column(Integer, nullable=False)

    chat_messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="turn",
    )

    __table_args__ = (UniqueConstraint("chat_id", "sequence_id", name="uq_chat_sequence"),)


class MessageType(enum.Enum):
    CHAT_MESSAGE = "chat_message"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"


class ChatMessage(BaseModel):
    __tablename__ = "chat_messages"

    message_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)

    turn_id = mapped_column(Uuid(as_uuid=True), ForeignKey("turns.turn_id"), nullable=False)
    turn = relationship("Turn", back_populates="messages")
    message_type: Mapped[MessageType]

    __mapper_args__ = {
        "polymorphic_on": "message_type",
        "polymorphic_identity": MessageType.CHAT_MESSAGE,
    }

    content = mapped_column(Text, nullable=False)


# A user message (usually a prompt) from the user.
class UserMessage(ChatMessage):
    __mapper_args__ = {
        "polymorphic_identity": MessageType.USER_MESSAGE,
    }


# An assistant message from a language model.
class AssistantMessage(ChatMessage):
    __mapper_args__ = {
        "polymorphic_identity": MessageType.ASSISTANT_MESSAGE,
    }
    # the model identifier for the language model that generated this message.
    # TODO(minqi): convert to a model relationship when we need it.
    assistant_model_name = mapped_column(String)


# TODO(minqi): Add comparison result (fka yupptake).
