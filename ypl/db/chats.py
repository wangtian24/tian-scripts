import enum
import uuid
from typing import TYPE_CHECKING, TypeVar

from fast_langdetect.ft_detect.infer import get_model_loaded
from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, JSON, Boolean, Column, Index, Text, UniqueConstraint, text
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.dialects.postgresql import ENUM as PostgresEnum
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlmodel import Field, Relationship

from ypl.backend.llm.moderation import ModerationReason
from ypl.db.base import BaseModel
from ypl.db.language_models import LanguageModel
from ypl.db.ratings import Category
from ypl.db.users import User

if TYPE_CHECKING:
    from ypl.db.chats import Category
    from ypl.db.rewards import Reward, RewardActionLog


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
    # Whether the chat is pinned by the user
    is_pinned: bool = Field(sa_column=Column(Boolean, nullable=False, server_default=text("false")))

    creator_user_id: str = Field(foreign_key="users.user_id", sa_type=Text)
    creator: User = Relationship(back_populates="chats")

    __table_args__ = (Index("ix_chats_created_at", text("created_at DESC")),)


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
        sa_relationship_kwargs={"lazy": "joined"},
    )

    evals: list["Eval"] = Relationship(back_populates="turn")

    creator_user_id: str = Field(foreign_key="users.user_id", nullable=False, sa_type=Text)
    creator: "User" = Relationship(back_populates="turns")

    turn_quality: "TurnQuality" = Relationship(back_populates="turn")

    reward_action_logs: list["RewardActionLog"] = Relationship(back_populates="turn")

    rewards: list["Reward"] = Relationship(back_populates="turn")

    __table_args__ = (UniqueConstraint("chat_id", "sequence_id", name="uq_chat_sequence"),)


# Note only names are used in the DB for enums.
# https://docs.sqlalchemy.org/en/20/core/type_basics.html#sqlalchemy.types.Enum
class MessageType(enum.Enum):
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    # The tl;dr feature (previously known as quick take)
    QUICK_RESPONSE_MESSAGE = "quick_response_message"


# Note only names are used in the DB for enums.
# https://docs.sqlalchemy.org/en/20/core/type_basics.html#sqlalchemy.types.Enum
class MessageUIStatus(enum.Enum):
    UNKNOWN = "unknown"
    # The user has seen the message.
    SEEN = "seen"
    # The user has dismissed the message.
    DISMISSED = "dismissed"
    # The user has selected the message.
    SELECTED = "selected"


class AssistantSelectionSource(enum.Enum):
    # For backwards compatibility.
    UNKNOWN = "unknown"
    # The user has selected the model manually.
    USER_SELECTED = "user_selected"
    # The model selection was fulfilled by the router.
    ROUTER_SELECTED = "router_selected"
    # A default model used for quick take.
    QUICK_TAKE_DEFAULT = "quick_take_default"


LanguageCodeType = TypeVar("LanguageCodeType", bound="LanguageCodeEnum")


class LanguageCodeEnum(enum.Enum):
    @classmethod
    def from_string(cls: type[LanguageCodeType], code: str) -> "LanguageCodeType | None":
        try:
            return cls[code.upper().strip()]
        except KeyError:
            return None


language_codes = sorted([label.replace("__label__", "") for label in get_model_loaded().get_labels()])

LanguageCode: type[LanguageCodeEnum] = enum.Enum(  # type: ignore
    "LanguageCode",
    {code.upper(): code for code in language_codes},
    type=LanguageCodeEnum,
)


class ModifierCategory(enum.Enum):
    length = "length"
    tone = "tone"
    style = "style"
    formatting = "formatting"
    complexity = "complexity"
    other = "other"


class PromptModifier(BaseModel, table=True):
    __tablename__ = "prompt_modifiers"

    prompt_modifier_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(index=True)
    category: ModifierCategory = Field(
        sa_column=Column(SQLAlchemyEnum(ModifierCategory), server_default=ModifierCategory.other.name)
    )
    text: str | None = Field(nullable=True)


class PromptModifierAssoc(BaseModel, table=True):
    __tablename__ = "prompt_modifier_assocs"

    assoc_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    prompt_modifier_id: uuid.UUID = Field(foreign_key="prompt_modifiers.prompt_modifier_id", index=True)
    chat_message_id: uuid.UUID = Field(foreign_key="chat_messages.message_id", index=True)

    __table_args__ = (UniqueConstraint("prompt_modifier_id", "chat_message_id", name="uq_prompt_modifier_assoc"),)


class ChatMessage(BaseModel, table=True):
    __tablename__ = "chat_messages"

    message_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    turn_id: uuid.UUID = Field(foreign_key="turns.turn_id", nullable=False, index=True)
    turn: Turn = Relationship(back_populates="chat_messages")
    message_type: MessageType = Field(sa_column=Column(SQLAlchemyEnum(MessageType), nullable=False))
    content: str = Field(nullable=False, sa_type=Text)
    # Deprecated: Use assistant_language_model_id instead.
    assistant_model_name: str | None = Field()
    content_tsvector: TSVECTOR | None = Field(default=None, sa_column=Column(TSVECTOR))
    content_pgvector: Vector | None = Field(default=None, sa_column=Column(Vector(1536)))
    category_id: uuid.UUID | None = Field(foreign_key="categories.category_id", nullable=True)
    category: "Category" = Relationship(back_populates="chat_messages")
    language_code: LanguageCode | None = Field(
        sa_column=Column(
            SQLAlchemyEnum(LanguageCode),
            nullable=True,
        )
    )

    # Streaming metrics for display purposes in the UI, such as average streaming speed.
    streaming_metrics: dict[str, str] = Field(default_factory=dict, sa_type=JSON, nullable=True)

    # The language model used to generate the message. Set when the message_type is assistant_message.
    assistant_language_model_id: uuid.UUID | None = Field(
        foreign_key="language_models.language_model_id", nullable=True
    )
    assistant_language_model: "LanguageModel" = Relationship(back_populates="chat_messages")

    # How the assitant language model was chosen.
    assistant_selection_source: AssistantSelectionSource = Field(
        sa_column=Column(
            SQLAlchemyEnum(AssistantSelectionSource),
            nullable=False,
            default=AssistantSelectionSource.UNKNOWN,
            server_default=AssistantSelectionSource.UNKNOWN.name,
        )
    )

    ui_status: MessageUIStatus = Field(
        sa_column=Column(
            SQLAlchemyEnum(MessageUIStatus),
            nullable=False,
            default=MessageUIStatus.UNKNOWN,
            server_default=MessageUIStatus.UNKNOWN.name,
        )
    )
    # When present, indicates in which order this message should be displayed in relation to
    # other messages in the same turn.
    turn_sequence_number: int | None = Field(nullable=True)

    # The prompt modifiers used to generate the message, using the association table.
    prompt_modifiers: list[PromptModifier] = Relationship(link_model=PromptModifierAssoc)

    # Needed for sa_type=JSON
    class Config:
        arbitrary_types_allowed = True

    message_evals: list["MessageEval"] = Relationship(back_populates="message", cascade_delete=True)


class EvalType(enum.Enum):
    # TODO: Will be deprecated after migration --> replaced by SELECTION.
    # Primitive evaluation of two responses where the user distributes 100 points between two responses.
    # The eval result is a dictionary where the key is the model name and the value is the number of points.
    SLIDER_V0 = "slider_v0"
    # TODO: Will be deprecated after migration --> replaced by QUICK_TAKE.
    # Thumbs up / thumbs down evaluation applied to a
    # single message. A positive value is thumbs up, and
    # negative value is thumbs down.
    THUMBS_UP_DOWN_V0 = "thumbs_up_down_v0"
    # TODO: Will be deprecated after migration --> replaced by QUICK_TAKE.
    # User-generated alternative to a Quick Take produced
    # by a model.
    QUICK_TAKE_SUGGESTION_V0 = "quick_take_suggestion_v0"
    # Eval where user selects one winner from multiple responses.
    SELECTION = "selection"
    # Eval where user indicates all responses are bad.
    ALL_BAD = "all_bad"
    # QuickTake eval can contain thumbs up/down and/or comment.
    QUICK_TAKE = "quick_take"


class MessageEval(BaseModel, table=True):
    __tablename__ = "message_evals"

    message_eval_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    message_id: uuid.UUID = Field(foreign_key="chat_messages.message_id", nullable=False, index=True)
    message: ChatMessage = Relationship(back_populates="message_evals")
    score: float = Field(nullable=True)
    user_comment: str | None = Field(nullable=True)
    eval_id: uuid.UUID = Field(foreign_key="evals.eval_id", nullable=False, index=True)
    eval: "Eval" = Relationship(back_populates="message_evals")


class Eval(BaseModel, table=True):
    __tablename__ = "evals"

    eval_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    user_id: uuid.UUID = Field(foreign_key="users.user_id", nullable=False, sa_type=Text)
    turn_id: uuid.UUID = Field(foreign_key="turns.turn_id", nullable=False, index=True)
    turn: Turn = Relationship(back_populates="evals")
    user: User = Relationship(back_populates="evals")
    eval_type: EvalType = Field(sa_column=Column(SQLAlchemyEnum(EvalType), nullable=False))
    user_comment: str | None = Field(nullable=True)
    judge_model_name: str | None = Field(nullable=True)
    message_evals: list[MessageEval] = Relationship(back_populates="eval", cascade_delete=True)
    reward_action_logs: list["RewardActionLog"] = Relationship(back_populates="eval")


class TurnQuality(BaseModel, table=True):
    __tablename__ = "turn_qualities"

    turn_quality_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    turn_id: uuid.UUID = Field(foreign_key="turns.turn_id", nullable=False, unique=True, index=True)
    turn: "Turn" = Relationship(back_populates="turn_quality", sa_relationship_kwargs={"lazy": "joined"})

    # The difficulty of the prompt: 1 (easy) to 10 (hard).
    # A difficult prompt requires increased capabilities from the LLM, including domain knowledge,
    # creativity, or increased cognitive load.
    prompt_difficulty: float | None = Field(nullable=True)

    # The novelty of the prompt: 1 (very similar to existing prompts) to 10 (unique and novel).
    # A novel prompt differs from known prompts in terms of structure, domain, or format.
    prompt_novelty: float | None = Field(nullable=True)

    # The contribution of the turn to the system: 1 (no contribution) to 10 (game-changing).
    # A "contributing" turn is one that results in changes in the state of the system, such as
    # modifications to the rankings of different language models.
    turn_contribution: float | None = Field(nullable=True)

    # The language model used to judge the difficulty of the prompt.
    prompt_difficulty_judge_model_id: uuid.UUID | None = Field(
        foreign_key="language_models.language_model_id", nullable=True
    )
    prompt_difficulty_judge_model: "LanguageModel" = Relationship(back_populates="turn_qualities")

    # Whether the prompt is safe according to a moderation model.
    prompt_is_safe: bool | None = Field(nullable=True)
    # The model used to moderate the prompt.
    prompt_moderation_model_name: str | None = Field(nullable=True)
    # If the prompt is not safe, the reasons for why it is not safe.
    prompt_unsafe_reasons: list[ModerationReason] | None = Field(
        sa_column=Column(ARRAY(PostgresEnum(ModerationReason)), nullable=True)
    )

    # The overall quality of the turn.
    quality: float | None = Field(nullable=True)

    def get_overall_quality(self) -> float | None:
        if self.prompt_is_safe is False:
            return 0

        # TODO(carmen): currently assumes all components are on the same scale, update with new scores
        components = [
            self.prompt_difficulty,
            self.prompt_novelty,
            self.turn_contribution,
        ]
        available_components = [c for c in components if c is not None]
        return sum(available_components) / len(available_components) if available_components else None


# TODO(minqi): Add comparison result (fka yupptake).
