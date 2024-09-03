import enum
import uuid
from enum import Enum

from sqlalchemy import Column, Text, UniqueConstraint
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlmodel import Field, ForeignKey, Relationship

from db.base import BaseModel
from db.users import User


# A chat can contain multiple conversations.
class Chat(BaseModel, table=True):
    __tablename__ = "chats"

    chat_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    title: str = Field(sa_column=Column(Text, nullable=False))
    # URL segment for the chat.
    # This field is world visible, do not leak information in this field.
    path: str = Field(sa_column=Column(Text, nullable=False, unique=True, index=True))
    turns: list["Turn"] = Relationship(
        back_populates="chat",
        sa_relationship_kwargs={"order_by": "Turn.sequence_id"},
    )

    # Whether the chat is public, which makes it visible in the feed.
    is_public: bool = Field(default=False, nullable=False)

    creator_user_id: str = Field(foreign_key="users.id", sa_type=Text)
    creator: User = Relationship(back_populates="chats")


class Turn(BaseModel, table=True):
    __tablename__ = "turns"

    turn_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)

    chat_id: uuid.UUID = Field(foreign_key="chats.chat_id", nullable=False)
    chat: "Chat" = Relationship(back_populates="turns")

    # Sequence of the turn in the chat. This sequence is not guaranteed to be
    # continuous, as messages may be deleted in the future.
    sequence_id: int = Field(nullable=False)

    chat_messages: list["ChatMessage"] = Relationship(
        back_populates="turn",
    )

    creator_user_id: str = Field(foreign_key="users.id", nullable=False, sa_type=Text)
    creator: "User" = Relationship(back_populates="turns")

    evals: list["Eval"] = Relationship(back_populates="turn")

    __table_args__ = (UniqueConstraint("chat_id", "sequence_id", name="uq_chat_sequence"),)


class MessageType(enum.Enum):
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    # The tl;dr feature (previously known as quick take)
    QUICK_RESPONSE_MESSAGE = "quick_response_message"


class ChatMessage(BaseModel, table=True):
    __tablename__ = "chat_messages"

    message_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    turn_id: uuid.UUID = Field(foreign_key="turns.turn_id", nullable=False)
    turn: Turn = Relationship(back_populates="chat_messages")
    message_type: MessageType = Field(sa_column=Column(SQLAlchemyEnum(MessageType), nullable=False))
    content: str = Field(nullable=False, sa_type=Text)
    assistant_model_name: str | None = Field()
    evals_as_message_1: list["Eval"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[Eval.message_1_id]",
            "primaryjoin": "ChatMessage.message_id == Eval.message_1_id",
        },
        back_populates="message_1",
    )
    evals_as_message_2: list["Eval"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[Eval.message_2_id]",
            "primaryjoin": "ChatMessage.message_id == Eval.message_2_id",
        },
        back_populates="message_2",
    )


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


class Eval(BaseModel, table=True):
    __tablename__ = "evals"

    eval_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    user_id: uuid.UUID = Field(foreign_key="users.id", nullable=False, sa_type=Text)
    user: User = Relationship(back_populates="evals")
    turn_id: uuid.UUID = Field(foreign_key="turns.turn_id", nullable=False)
    turn: Turn = Relationship(back_populates="evals")
    eval_type: EvalType = Field(sa_column=Column(SQLAlchemyEnum(EvalType), nullable=False))
    message_1_id: uuid.UUID = Field(sa_column=Column(ForeignKey("chat_messages.message_id"), nullable=False))
    message_1: "ChatMessage" = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[Eval.message_1_id]",
            "primaryjoin": "Eval.message_1_id == ChatMessage.message_id",
        },
        back_populates="evals_as_message_1",
    )

    message_2_id: uuid.UUID | None = Field(sa_column=Column(ForeignKey("chat_messages.message_id"), nullable=True))
    message_2: "ChatMessage" = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[Eval.message_2_id]",
            "primaryjoin": "Eval.message_2_id == ChatMessage.message_id",
        },
        back_populates="evals_as_message_2",
    )
    score_1: float | None = Field(nullable=True)
    score_2: float | None = Field(nullable=True)
    user_comment: str | None = Field(nullable=True)


# TODO(minqi): Add comparison result (fka yupptake).
