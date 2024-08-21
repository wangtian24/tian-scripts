import enum
import uuid
from enum import Enum

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import BaseModel
from db.users import User


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

    creator_user_id = mapped_column(Text, ForeignKey("users.id", name="chat_creator_user_id_fkey"), nullable=False)
    creator = relationship("User", back_populates="chats")


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

    creator_user_id = mapped_column(Text, ForeignKey("users.id", name="turn_creator_user_id_fkey"), nullable=False)
    creator = relationship("User", back_populates="turns")

    evals: Mapped[list["Eval"]] = relationship(back_populates="turn")

    __table_args__ = (UniqueConstraint("chat_id", "sequence_id", name="uq_chat_sequence"),)


class MessageType(enum.Enum):
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
        "polymorphic_abstract": True,
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


class EvalType(Enum):
    # Primitive evaluation of two responses where the user distributes 100 points between two responses.
    # The eval result is a dictionary where the key is the model name and the value is the number of points.
    SLIDER_V0 = "slider_v0"
    # Thumbs up / thumbs down evaluation applied to a
    # single message. A positive value is thumbs up, and
    # negative value is thumbs down.
    THUMBS_UP_DOWN_V0 = "thumbs_up_down_v0"
    # User-generated alternative to a Quick Take produced
    # by a model.
    QUICK_TAKE_SUGGESTION_V0 = "quick_take_suggestion_v0"


class Eval(BaseModel):
    __tablename__ = "evals"

    eval_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    user: Mapped[User] = relationship()
    turn_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("turns.turn_id"), nullable=False)
    turn: Mapped[Turn] = relationship()
    eval_type: Mapped[EvalType] = mapped_column(nullable=False)
    message_1_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_messages.message_id"), nullable=False)
    message_1: Mapped[ChatMessage] = relationship()
    message_2_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_messages.message_id"), nullable=True)
    message_2: Mapped[ChatMessage] = relationship()
    score_1: Mapped[float] = mapped_column(nullable=True)
    score_2: Mapped[float] = mapped_column(nullable=True)
    user_comment: Mapped[str] = mapped_column(nullable=True)


# TODO(minqi): Add comparison result (fka yupptake).
