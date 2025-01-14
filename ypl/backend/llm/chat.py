import asyncio
import logging
import time
import traceback
from collections import defaultdict
from collections.abc import Generator, Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar
from uuid import UUID

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import Insert as pg_insert
from sqlalchemy.orm import joinedload
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.config import settings
from ypl.backend.db import get_async_engine
from ypl.backend.llm.constants import ACTIVE_MODELS_BY_PROVIDER, ChatProvider
from ypl.backend.llm.db_helpers import (
    deduce_original_providers,
    get_chat_history,
    get_chat_model,
    get_preferences,
    get_shown_models,
    get_user_message,
)
from ypl.backend.llm.labeler import QT_CANT_ANSWER, MultiLLMLabeler, QuickTakeGenerator
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.prompt_selector import CategorizedPromptModifierSelector, get_modifiers_by_model, store_modifiers
from ypl.backend.llm.provider.provider_clients import get_model_provider_tuple
from ypl.backend.llm.routing.debug import RoutingDebugInfo, build_routing_debug_info
from ypl.backend.llm.routing.route_data_type import PreferredModel, RoutingPreference
from ypl.backend.llm.routing.router_state import RouterState
from ypl.backend.llm.vendor_langchain_adapter import GeminiLangChainAdapter, OpenAILangChainAdapter
from ypl.backend.prompts import ALL_MODELS_IN_CHAT_HISTORY_PREAMBLE, RESPONSE_SEPARATOR
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.monitoring import metric_inc, metric_inc_by, metric_record
from ypl.db.attachments import Attachment
from ypl.db.chats import (
    AssistantSelectionSource,
    Chat,
    ChatMessage,
    CompletionStatus,
    MessageModifierStatus,
    MessageType,
    MessageUIStatus,
    PromptModifier,
    PromptModifierAssoc,
    Turn,
)
from ypl.db.language_models import LanguageModel
from ypl.db.redis import get_upstash_redis_client
from ypl.utils import tiktoken_trim

MAX_LOGGED_MESSAGE_LENGTH = 200
DEFAULT_HIGH_SIM_THRESHOLD = 0.825
DEFAULT_UNIQUENESS_THRESHOLD = 0.75
YuppMessage = HumanMessage | AIMessage | SystemMessage  # this is needed for proper Pydantic typecasting
YuppMessageRow = list[YuppMessage]


class SelectIntent(str, Enum):
    NEW_CHAT = "new_chat"
    NEW_TURN = "new_turn"
    SHOW_ME_MORE = "show_me_more"


class SelectModelsV2Request(BaseModel):
    user_id: str | None = None
    intent: SelectIntent
    prompt: str | None = None  # prompt to use for routing
    num_models: int = 2  # number of models to select
    required_models: list[str] | None = None  # models selected explicitly by the user
    chat_id: str | None = None  # chat ID to use for routing
    turn_id: str | None = None  # turn ID to use for routing
    provided_categories: list[str] | None = None  # categories provided by the user
    debug_level: int = 0  # 0: return no debug info, log only, 1: return debug


class SelectModelsV2Response(BaseModel):
    models: list[tuple[str, list[tuple[str, str]]]]  # list of (model, list[(prompt modifier ID, prompt modifier)])
    provider_map: dict[str, str]  # map from model to provider
    fallback_models: list[tuple[str, list[tuple[str, str]]]]  # list of fallback models and modifiers
    routing_debug_info: RoutingDebugInfo | None = None


async def select_models_plus(request: SelectModelsV2Request) -> SelectModelsV2Response:
    from ypl.backend.llm.routing.router import get_simple_pro_router

    async def select_models_(
        required_models: list[str] | None = None,
        show_me_more_models: list[str] | None = None,
        provided_categories: list[str] | None = None,
    ) -> tuple[RouterState, RouterState]:
        num_models = request.num_models

        # select N models
        router = await get_simple_pro_router(
            prompt,
            num_models,
            preference,
            user_selected_models=required_models,
            show_me_more_models=show_me_more_models,
            provided_categories=provided_categories,
            extra_prefix="PRIMARY-",
        )
        all_models_state = RouterState.new_all_models_state()
        selected_models_rs = router.select_models(state=all_models_state)

        # select N more models as fallback, not same as first N
        router = await get_simple_pro_router(
            prompt,
            num_models,
            preference,
            # do not include user selected models already used
            user_selected_models=[m for m in (required_models or []) if m not in selected_models_rs.selected_models],
            show_me_more_models=show_me_more_models,
            provided_categories=provided_categories,
            extra_prefix="FALLBACK-",
        )
        all_fallback_models = RouterState.new_all_models_state()
        all_fallback_models = all_fallback_models.emplaced(
            # just keep the models not already in the return models (any remainders), select within them
            all_models=all_fallback_models.all_models.difference(selected_models_rs.get_sorted_selected_models())
        )
        fallback_models_rs = router.select_models(state=all_fallback_models)

        return selected_models_rs, fallback_models_rs

    metric_inc(f"routing/intent_{request.intent}")
    start_time = time.time()

    match request.intent:
        case SelectIntent.NEW_CHAT | SelectIntent.NEW_TURN:
            assert request.prompt is not None, "prompt is required for NEW_CHAT or NEW_TURN intent"
            prompt = request.prompt
        case SelectIntent.SHOW_ME_MORE:
            assert request.turn_id is not None, "turn_id is required for SHOW_ME_MORE intent"
            prompt = get_user_message(request.turn_id)

    match request.intent:
        case SelectIntent.NEW_TURN:
            preference, user_selected_models = get_preferences(request.user_id, request.chat_id)  # type: ignore[arg-type]
            request.required_models = list(dict.fromkeys(user_selected_models + (request.required_models or [])))
        case SelectIntent.SHOW_ME_MORE:
            preference, user_selected_models = get_preferences(request.user_id, request.chat_id)  # type: ignore[arg-type]
            preference.turns = preference.turns or []
            preference.turns.append(PreferredModel(models=user_selected_models, preferred=None))
            request.required_models = []
        case _:
            preference = RoutingPreference(turns=[], user_id=request.user_id)

    preference.debug_level = request.debug_level
    show_me_more_models = []

    if request.intent == SelectIntent.SHOW_ME_MORE:
        shown_models = get_shown_models(request.turn_id)  # type: ignore[arg-type]

        if preference.turns is None:
            preference.turns = []

        show_me_more_models = shown_models[-request.num_models :]
        preference.turns.append(PreferredModel(models=list(dict.fromkeys(shown_models)), preferred=None))

    if request.intent == SelectIntent.NEW_TURN and preference.turns and not preference.turns[-1].has_evaluation:
        models = list(dict.fromkeys(preference.turns[-1].models + (request.required_models or [])))
    else:
        models = request.required_models or []

    selected_models_rs, fallback_models_rs = await select_models_(
        required_models=models,
        show_me_more_models=show_me_more_models,
        provided_categories=request.provided_categories,
    )

    # TODO(tian) - redundant??
    # models = models[: request.num_models]
    # fallback_models = fallback_models[: request.num_models]

    selected_models = selected_models_rs.get_sorted_selected_models()
    fallback_models = fallback_models_rs.get_sorted_selected_models()

    prompt_modifiers: dict[str, list[tuple[str, str]]] = {}
    if request.intent != SelectIntent.NEW_CHAT:
        try:
            selector = CategorizedPromptModifierSelector.make_default_from_db()

            if request.chat_id:
                modifier_history = get_modifiers_by_model(request.chat_id)
            else:
                modifier_history = {}

            prompt_modifiers = selector.select_modifiers(
                selected_models + fallback_models,
                modifier_history,
                selected_models_rs.applicable_modifiers,
            )

            if request.turn_id:
                asyncio.create_task(store_modifiers(request.turn_id, prompt_modifiers))
        except Exception as e:
            logging.error(f"Error selecting modifiers: {e}")

    # increment counters
    metric_inc_by("routing/count_models_served", len(models))
    if len(models) > 0:
        metric_inc(f"routing/count_first_{models[0]}")
    if len(models) > 1:
        metric_inc(f"routing/count_second_{models[1]}")
    for model in models:
        metric_inc(f"routing/count_chosen_{model}")
    metric_record(f"routing/latency_{request.intent}_ms", int((time.time() - start_time) * 1000))

    routing_debug_info: RoutingDebugInfo = build_routing_debug_info(
        selected_models_rs=selected_models_rs,
        fallback_models_rs=fallback_models_rs,
        required_models=request.required_models,
    )

    if logging.root.getEffectiveLevel() == logging.DEBUG:
        for model, model_debug in routing_debug_info.model_debug.items():
            is_selected = " S => " if model in selected_models else ""
            logging.debug(f"> {model:<50}{is_selected:<8}{model_debug.score:-10.1f}{model_debug.journey}")

    return SelectModelsV2Response(
        models=[(model, prompt_modifiers.get(model, [])) for model in selected_models],
        fallback_models=[(model, prompt_modifiers.get(model, [])) for model in fallback_models],
        provider_map=deduce_original_providers(tuple(models)),
        routing_debug_info=routing_debug_info if request.debug_level > 0 else None,
    )


def get_chat_history_model(
    info: ModelInfo,
    chat_model_pool: dict[ChatProvider, list[str]] = ACTIVE_MODELS_BY_PROVIDER,
    **chat_kwargs: Any | None,
) -> BaseChatModel:
    llm = get_chat_model(info, chat_model_pool=chat_model_pool, **chat_kwargs)
    conv_template = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return conv_template | llm  # type: ignore


# langchain uses Pydantic v1 in YuppMessage; using for compatibility
class Persona(BaseModel):
    persona: str = ""
    interests: list[str] = []
    style: str = ""

    def __hash__(self) -> int:
        return hash((self.persona, tuple(self.interests), self.style))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Persona):
            return False

        return (
            self.persona == other.persona
            and tuple(self.interests) == tuple(other.interests)
            and self.style == other.style
        )


# langchain uses Pydantic v1 in YuppMessage; using for compatibility
class YuppChatMessageHistory(BaseModel):
    """
    Holds the chat history for a Yupp chat. Each turn can be composed of multiple chat messages (e.g., from two
    LLMs in parallel), so we use a list of messages to represent a turn.
    """

    messages: list[YuppMessageRow] = []
    judgements: list[int | None] = []  # for v1, it is assumed that all judgements are between 1 and 100 inclusive
    eval_llms: list[str] = []
    judge_llm: str | None = None
    user_persona: Persona | None = None
    chat_id: str | None = None

    def initial_prompt_and_responses(self) -> tuple[str | None, Any, list[Any]]:
        """Returns the prompt and respones from the initial turn."""
        return self.chat_id, self.messages[0][0].content, [m.content for m in self.messages[1]]

    def triplet_blocks(self) -> Generator[tuple[YuppMessage, YuppMessage, YuppMessage], None, None]:
        """Generates triplet blocks of user-llm1-llm2 messages, similar to the front-end's behavior."""
        for idx in range(0, (len(self.messages) // 2) * 2, 2):
            if len(self.messages[idx]) != 1 or len(self.messages[idx + 1]) != 2:
                raise ValueError("Each block must have one user message and two LLM messages")

            yield self.messages[idx][0], self.messages[idx + 1][0], self.messages[idx + 1][1]


class MultiChatUser:
    """
    Represents a conversational agent capable of responding to one or more messages simultaneously. Keeps track of the
    chat history and various attributes. Each context is associated with a unique chat history.
    """

    def __init__(self) -> None:
        self.chat_history: ChatMessageHistory | None = None

    def copy(self) -> "MultiChatUser":
        """Creates a copy of the chat user."""
        raise NotImplementedError

    @property
    def last_message(self) -> YuppMessage:
        """Returns the last generated message from the synthetic user."""
        assert self.chat_history is not None, "Must be called within the context"
        return self.chat_history.messages[-1]  # type: ignore

    def reset(self) -> None:
        self.chat_history = ChatMessageHistory()

    async def areset(self) -> None:
        self.chat_history = ChatMessageHistory()

    async def __aenter__(self) -> "MultiChatUser":
        await self.areset()
        return self

    def __enter__(self) -> "MultiChatUser":
        self.reset()
        return self

    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        self.reset()

    async def __aexit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        await self.areset()

    def respond(self, *messages: YuppMessage) -> YuppMessage:
        """Responds to messages from one or more LLMs."""
        assert self.chat_history is not None, "Chat history not set. Did you forget to enter the context?"
        return self._respond(*messages)

    def _respond(self, *messages: YuppMessage) -> YuppMessage:
        raise NotImplementedError

    async def arespond(self, *messages: YuppMessage) -> YuppMessage:
        """Responds to a message asynchronously."""
        assert self.chat_history is not None, "Chat history not set. Did you forget to enter the context?"
        return await self._arespond(*messages)

    async def _arespond(self, *messages: YuppMessage) -> YuppMessage:
        raise NotImplementedError


class LLMChatAssistant(MultiChatUser):
    def __init__(self, llm: BaseChatModel):
        super().__init__()
        self.llm = llm

    def _respond(self, *messages: YuppMessage) -> YuppMessage:
        """Responds to the first message only"""
        assert len(messages) == 1, "Only one message is supported"
        assert self.chat_history is not None

        message = messages[0]
        response = self.llm.invoke(
            dict(input=message.content, chat_history=self.chat_history.messages)  # type: ignore
        )
        self.chat_history.messages.append(message)
        self.chat_history.messages.append(response)

        return response  # type: ignore

    async def _arespond(self, *messages: YuppMessage) -> YuppMessage:
        """Responds to the first message only"""
        assert len(messages) == 1, "Only one message is supported"
        assert self.chat_history is not None

        message = messages[0]
        response = await self.llm.ainvoke(
            dict(input=message.content, chat_history=self.chat_history.messages)  # type: ignore
        )
        self.chat_history.messages.append(message)
        self.chat_history.messages.append(response)

        return response  # type: ignore


ChatUserType = TypeVar("ChatUserType", bound=MultiChatUser)


class YuppChatUserGenerator(Generic[ChatUserType]):
    """Generates chat users."""

    async def agenerate_users(self) -> Generator[ChatUserType, None, None]:
        """Generates chat users asynchronously. Defaults to synchronous implementation if not overriden."""
        return self.generate_users()

    def generate_users(self) -> Generator[ChatUserType, None, None]:
        """Generates chat users."""
        raise NotImplementedError


class YuppChatIO:
    def append_chat(self, chat: YuppChatMessageHistory) -> "YuppChatIO":
        """Appends a chat to the writer."""
        raise NotImplementedError

    def write_all_chats(self, chats: list[YuppChatMessageHistory]) -> "YuppChatIO":
        for chat in chats:
            self.append_chat(chat)

        return self

    def read_chats(self) -> list[YuppChatMessageHistory]:
        """Reads chats from the writer."""
        raise NotImplementedError

    def delete(self) -> None:
        """Deletes the object underlying the writer."""
        raise NotImplementedError

    def flush(self) -> None:
        """Flushes the writer."""
        pass


class JsonChatIO(YuppChatIO):
    def __init__(self, filename: str) -> None:
        self.path = Path(filename)
        self.chats: list[YuppChatMessageHistory] = []

    def append_chat(self, chat: YuppChatMessageHistory) -> "JsonChatIO":
        self.chats.append(chat)
        return self

    def read_chats(self) -> list[YuppChatMessageHistory]:
        chats = []

        with self.path.open() as f:
            for line in f:
                chats.append(YuppChatMessageHistory.parse_raw(line))

        self.chats = chats
        return self.chats

    def delete(self) -> None:
        self.path.unlink()

    def flush(self, mode: str = "a") -> None:
        with self.path.open(mode=mode) as f:
            for chat in self.chats:
                f.write(chat.json())
                f.write("\n")

        self.chats = []


ChatMessageType1 = TypeVar("ChatMessageType1", bound=BaseMessage)
ChatMessageType2 = TypeVar("ChatMessageType2", bound=BaseMessage)


def chat_message_cast_to(message: ChatMessageType1, target_type: type[ChatMessageType2]) -> ChatMessageType2:
    message.type = target_type.schema()["properties"]["type"]["default"]
    return target_type(**message.dict())


def get_db_message_type(message: ChatMessageType1) -> MessageType:
    match message:
        case HumanMessage():
            return MessageType.USER_MESSAGE
        case AIMessage():
            return MessageType.ASSISTANT_MESSAGE
        case _:
            raise ValueError(f"Unsupported message type: {type(message)}")


async def get_turn_id_from_message_id(message_id: UUID) -> UUID | None:
    """Get turn_id from message_id by querying the chat_messages table."""
    async with AsyncSession(get_async_engine()) as session:
        message = await session.get(ChatMessage, message_id)
        if message:
            return message.turn_id
        return None


def _get_assistant_messages(
    turn_messages: list[ChatMessage],
    model: str,
    use_all_models_in_chat_history: bool,
) -> list[BaseMessage]:
    """Get assistant messages for a turn.

    If use_all_models_in_chat_history is True, includes assistant messages from all models, indicating which ones
    are from the current model and which one was preferred by the user (if any).
    If use_all_models_in_chat_history is False, includes only the preferred messages, or the first message if none
    were selected.
    """
    messages: list[BaseMessage] = []
    assistant_msgs = [msg for msg in turn_messages if msg.message_type == MessageType.ASSISTANT_MESSAGE]
    if not assistant_msgs:
        return messages

    if use_all_models_in_chat_history and len(assistant_msgs) > 1:
        all_content = []
        for msg in assistant_msgs:
            content = msg.content or ""
            if msg.assistant_language_model.internal_name == model:
                # A previous response from the current assistant.
                if content:
                    content = "This was your response:\n\n" + content
                else:
                    content = "(Your response was empty)"
            else:
                # A previous response from another assistant.
                if content:
                    # Only include responses from other assistants if non-empty.
                    content = "A response from another assistant:\n\n" + content
            if msg.ui_status == MessageUIStatus.SELECTED and content:
                content += "\n\n(This response was preferred by the user)"
            if content:
                all_content.append(content)
        if all_content:
            content = ALL_MODELS_IN_CHAT_HISTORY_PREAMBLE + RESPONSE_SEPARATOR.join(all_content)
            messages.append(AIMessage(content=content))
    else:
        selected_msg = next(
            (msg for msg in assistant_msgs if msg.ui_status == MessageUIStatus.SELECTED),
            None,
        )
        if not selected_msg:
            selected_msg = next(
                (msg for msg in assistant_msgs if msg.assistant_language_model.internal_name == model),
                assistant_msgs[0],  # Fallback to first message if none selected
            )
        if selected_msg:
            content = selected_msg.content
        else:
            content = None
            log_info = {
                "message": "No selected message in turn",
                "model_for_selected_message_lookup": model,
                "turn_id": assistant_msgs[0].turn_id,
            }
            logging.warning(json_dumps(log_info))

        # if content is null, a place holder is added as part of sanitize_messages.py/replace_empty_messages()
        messages.append(AIMessage(content=content))

    return messages


def _get_enhanced_user_message(messages: list[ChatMessage]) -> HumanMessage:
    user_msgs = [msg for msg in messages if msg.message_type == MessageType.USER_MESSAGE]
    if not user_msgs:
        raise ValueError("No user messages found")
    if len(user_msgs) > 1:
        raise ValueError("Multiple user messages found")
    user_msg = user_msgs[0]
    attachments = user_msg.attachments or []
    return HumanMessage(
        content=user_msg.content,
        additional_kwargs={"attachments": attachments},
    )


async def get_curated_chat_context(
    chat_id: UUID,
    use_all_models_in_chat_history: bool,
    model: str,
    current_turn_id: UUID | None = None,
) -> list[BaseMessage]:
    """Fetch chat history and format it for OpenAI context.

    Args:
        chat_id: The chat ID to fetch history for.
        use_all_models_in_chat_history: Whether to include all models in the chat history.
        model: The model to fetch history for.
        current_turn_id: The current turn ID.
    """
    query = (
        select(ChatMessage)
        .join(Turn, Turn.turn_id == ChatMessage.turn_id)  # type: ignore[arg-type]
        .join(Chat, Chat.chat_id == Turn.chat_id)  # type: ignore[arg-type]
        .outerjoin(Attachment, Attachment.chat_message_id == ChatMessage.message_id)  # type: ignore[arg-type]
        .options(
            joinedload(ChatMessage.assistant_language_model).load_only(LanguageModel.internal_name),  # type: ignore
            joinedload(ChatMessage.attachments),  # type: ignore
        )
        .where(
            Chat.chat_id == chat_id,
            ChatMessage.deleted_at.is_(None),  # type: ignore[union-attr]
            Turn.deleted_at.is_(None),  # type: ignore[union-attr]
            Chat.deleted_at.is_(None),  # type: ignore[union-attr]
            Turn.turn_id != current_turn_id,
        )
        .order_by(
            Turn.sequence_id.asc(),  # type: ignore[attr-defined]
            ChatMessage.turn_sequence_number.asc(),  # type: ignore[union-attr]
        )
    )

    formatted_messages: list[BaseMessage] = []
    async with AsyncSession(get_async_engine()) as session:
        result = await session.exec(query)
        messages = result.unique().all()

        # Group messages by turn_id
        turns: defaultdict[UUID, list[ChatMessage]] = defaultdict(list)
        for msg in messages:
            turns[msg.turn_id].append(msg)

        for turn_messages in turns.values():
            # Get user messages
            formatted_messages.append(_get_enhanced_user_message(turn_messages))
            # Get assistant messages
            formatted_messages.extend(_get_assistant_messages(turn_messages, model, use_all_models_in_chat_history))

    info = {
        "message": "chat_context",
        "chat_id": str(chat_id),
        "model": model,
    }
    for i, fmsg in enumerate(formatted_messages):
        info[f"message_{i}_content"] = (
            str(fmsg.content[:MAX_LOGGED_MESSAGE_LENGTH]) + "..."
            if len(fmsg.content) > MAX_LOGGED_MESSAGE_LENGTH
            else str(fmsg.content)
        )
        info[f"message_{i}_type"] = type(fmsg).__name__
    logging.info(json_dumps(info))

    return formatted_messages


class Intent(Enum):
    EAGER_PERSIST = "eager_persist"
    FINAL_PERSIST = "final_persist"


# TODO(bhanu) - add retry logic -
async def upsert_chat_message(
    intent: Intent,
    turn_id: UUID,
    message_id: UUID,
    model: str,
    message_type: MessageType,
    turn_seq_num: int,
    assistant_selection_source: AssistantSelectionSource,
    prompt_modifier_ids: list[UUID] | None,
    content: str | None = "",  # DB doesn't accept null value, defaulting to empty string.
    streaming_metrics: dict[str, str] | None = None,
    message_metadata: dict[str, str] | None = None,
    completion_status: CompletionStatus | None = None,
    modifier_status: MessageModifierStatus | None = None,
) -> None:
    """
    We have split the columns into two parts.
    items that are available before streaming - turn_id, message_id, message_type, turn_sequence_number,
    assistant_model_name,assistant_language_model_id, assistant_selection_source, message_metadata (etc)
    items that are available after streaming - content, streaming_metrics, message_metadata (etc)
    Items #1 are eager persisted.
    Items #2 are persisted after streaming (while addressing conflict and potential race condition)
    Please respect the above convention for future enhancements.

    """
    result = get_model_provider_tuple(model)

    if result is None:
        raise ValueError(f"No model and provider found for {model}")
    language_model = result[0]

    async with AsyncSession(get_async_engine()) as session:
        try:
            # Prepare values for upsert
            values = {
                "turn_id": turn_id,
                "message_id": message_id,
                "message_type": message_type,
                "content": content,
                "assistant_model_name": model,
                "streaming_metrics": streaming_metrics or {},
                "turn_sequence_number": turn_seq_num,
                "assistant_language_model_id": language_model.language_model_id,
                "assistant_selection_source": assistant_selection_source,
                "message_metadata": message_metadata,
                "completion_status": completion_status,
                "modifier_status": modifier_status,
            }

            # Perform upsert using ON CONFLICT for ChatMessage
            stmt = pg_insert(ChatMessage).values(**values)
            # For the edge case that Eager_Persist faces conflict,
            # it implies that final_persist has already been executed, so do_nothing, all fields already persisted.
            if intent == Intent.EAGER_PERSIST:
                stmt = stmt.on_conflict_do_nothing()
            else:
                stmt = stmt.on_conflict_do_update(
                    index_elements=["message_id"],
                    set_={
                        "content": stmt.excluded.content,
                        "streaming_metrics": stmt.excluded.streaming_metrics,
                        "message_metadata": stmt.excluded.message_metadata,
                        "completion_status": stmt.excluded.completion_status,
                        "modified_at": func.current_timestamp(),
                        "modifier_status": stmt.excluded.modifier_status,
                    },
                )
            await session.exec(stmt)  # type: ignore[call-overload]

            # Insert prompt modifier associations with ON CONFLICT DO NOTHING
            if prompt_modifier_ids:
                assoc_values = [
                    {"prompt_modifier_id": modifier_id, "chat_message_id": message_id}
                    for modifier_id in prompt_modifier_ids
                ]
                assoc_stmt = pg_insert(PromptModifierAssoc).values(assoc_values)
                # if it's already present, no need to update any new fields, do nothing.
                assoc_stmt = assoc_stmt.on_conflict_do_nothing()
                await session.exec(assoc_stmt)  # type: ignore[call-overload]

            await session.commit()

        except Exception as e:
            await session.rollback()
            logging.error(f"Error upserting chat message: {str(e)} \n" + traceback.format_exc())
            raise


async def get_active_prompt_modifiers() -> list[PromptModifier]:
    async with AsyncSession(get_async_engine()) as session:
        result = await session.execute(select(PromptModifier).where(PromptModifier.deleted_at.is_(None)))  # type: ignore
        return result.scalars().all()  # type: ignore


# Models to use if no specific model was requested.
MODELS_FOR_DEFAULT_QT = ["gpt-4o", "gpt-4o-mini", "gemini-2.0-flash-exp"]
# Model to use while supplying only the prompts from the chat history, instead of the full chat history.
MODEL_FOR_PROMPT_ONLY = "gpt-4o"
MODEL_FOR_PROMPT_ONLY_FULL_NAME = MODEL_FOR_PROMPT_ONLY + ":prompt-only"
# Fine-tuned model to use that minimizes truncations and formatting in responses
# More details at https://platform.openai.com/finetune/ftjob-VupgrOxNp0ApGhGKDgspdGjb
MODEL_FOR_FINETUNE_QT = "gpt-4o"
MODEL_FOR_FINETUNE_QT_FULL_NAME = "ft:gpt-4o-2024-08-06:yupp::AgJJZBsG"

GPT_4O_MINI_LLM = None
GPT_4O_LLM = None
FINE_TUNED_GPT_4O_LLM = None
GEMINI_15_FLASH_LLM = None
GEMINI_2_FLASH_LLM = None


def get_gpt_4o_mini_llm() -> OpenAILangChainAdapter:
    global GPT_4O_MINI_LLM
    if GPT_4O_MINI_LLM is None:
        GPT_4O_MINI_LLM = OpenAILangChainAdapter(
            model_info=ModelInfo(
                provider=ChatProvider.OPENAI,
                model="gpt-4o-mini",
                api_key=settings.OPENAI_API_KEY,
            ),
            model_config_=dict(
                temperature=0.0,
                max_tokens=40,
            ),
        )
    return GPT_4O_MINI_LLM


def get_gpt_4o_llm() -> OpenAILangChainAdapter:
    global GPT_4O_LLM
    if GPT_4O_LLM is None:
        GPT_4O_LLM = OpenAILangChainAdapter(
            model_info=ModelInfo(
                provider=ChatProvider.OPENAI,
                model="gpt-4o",
                api_key=settings.OPENAI_API_KEY,
            ),
            model_config_=dict(
                temperature=0.0,
                max_tokens=40,
            ),
        )
    return GPT_4O_LLM


def get_fine_tuned_gpt_4o_llm() -> OpenAILangChainAdapter:
    global FINE_TUNED_GPT_4O_LLM
    if FINE_TUNED_GPT_4O_LLM is None:
        FINE_TUNED_GPT_4O_LLM = OpenAILangChainAdapter(
            model_info=ModelInfo(
                provider=ChatProvider.OPENAI,
                model=MODEL_FOR_FINETUNE_QT_FULL_NAME,
                api_key=settings.OPENAI_API_KEY,
            ),
            model_config_=dict(
                temperature=0.0,
                max_tokens=40,
            ),
        )
    return FINE_TUNED_GPT_4O_LLM


def get_gemini_15_flash_llm() -> GeminiLangChainAdapter:
    global GEMINI_15_FLASH_LLM
    if GEMINI_15_FLASH_LLM is None:
        GEMINI_15_FLASH_LLM = GeminiLangChainAdapter(
            model_info=ModelInfo(
                provider=ChatProvider.GOOGLE,
                model="gemini-1.5-flash-002",
                api_key=settings.GOOGLE_API_KEY,
            ),
            model_config_=dict(
                project_id=settings.GCP_PROJECT_ID,
                region=settings.GCP_REGION,
                temperature=0.0,
                max_output_tokens=40,
                top_k=1,
            ),
        )
    return GEMINI_15_FLASH_LLM


def get_gemini_2_flash_llm() -> GeminiLangChainAdapter:
    global GEMINI_2_FLASH_LLM
    if GEMINI_2_FLASH_LLM is None:
        GEMINI_2_FLASH_LLM = GeminiLangChainAdapter(
            model_info=ModelInfo(
                provider=ChatProvider.GOOGLE,
                model="gemini-2.0-flash-exp",
                api_key=settings.GOOGLE_API_KEY,
            ),
            model_config_=dict(
                project_id=settings.GCP_PROJECT_ID,
                region=settings.GCP_REGION_GEMINI_2,
                temperature=0.0,
                max_output_tokens=40,
                top_k=1,
            ),
        )
    return GEMINI_2_FLASH_LLM


QT_LLMS = None
QT_MAX_CONTEXT_LENGTH = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 16000,
    "gemini-1.5-flash-002": 1000000,
    "gemini-2.0-flash-exp": 1000000,
}


def get_qt_llms() -> Mapping[str, BaseChatModel]:
    global QT_LLMS
    if QT_LLMS is None:
        QT_LLMS = {
            "gpt-4o": get_gpt_4o_llm(),
            "gpt-4o-mini": get_gpt_4o_mini_llm(),
            MODEL_FOR_FINETUNE_QT_FULL_NAME: get_fine_tuned_gpt_4o_llm(),
            "gemini-1.5-flash-002": get_gemini_15_flash_llm(),
            "gemini-2.0-flash-exp": get_gemini_2_flash_llm(),
        }
    return QT_LLMS


class QuickTakeResponse(BaseModel):
    quicktake: str
    model: str


class QuickTakeRequest(BaseModel):
    prompt: str | None = None
    model: str | None = None  # one of the entries in QT_LLMS; if none, use MODELS_FOR_DEFAULT_QT
    timeout_secs: float = settings.DEFAULT_QT_TIMEOUT_SECS


class PromptModifierInfo(BaseModel):
    prompt_modifier_id: str
    name: str
    description: str | None = None


def get_quicktake_generator(
    model: str,
    chat_history: list[dict[str, Any]],
    prompt_only: bool = False,
    timeout_secs: float = settings.DEFAULT_QT_TIMEOUT_SECS,
) -> QuickTakeGenerator:
    """Get a quicktake generator for a given model, or raise if the model is not supported."""
    if prompt_only:
        # Use only the prompts from the chat history.
        chat_history = [m for m in chat_history if m["role"] == "user"]
    return QuickTakeGenerator(get_qt_llms()[model], chat_history, timeout_secs=timeout_secs)


async def generate_quicktake(
    request: QuickTakeRequest,
    chat_id: str | None = None,
    turn_id: str | None = None,
    chat_history: list[dict[str, Any]] | None = None,
) -> QuickTakeResponse:
    """
    Generates a quicktake for a given chat_id or chat_history. If chat_history is provided, it will be used instead of
    chat_id and turn_id.

    Args:
        chat_id: The chat ID to fetch history for.
        turn_id: The turn ID to fetch history for.
        chat_history: The chat history to use.
    """
    match chat_id, chat_history:
        case None, None:
            raise ValueError("Either chat_id or chat_history must be provided")
        case None, _:
            pass
        case _, None:
            assert chat_id is not None  # because mypy cannot infer this
            chat_history = get_chat_history(chat_id, turn_id)

    assert chat_history is not None, "chat_history is null"

    response_model = ""
    timeout_secs = request.timeout_secs
    start_time = time.time()
    responses_by_model: dict[str, str] = {}

    try:
        if not request.model:
            # Default: use multiple models
            labelers: dict[str, Any] = {
                model: get_quicktake_generator(model, chat_history, timeout_secs=timeout_secs)
                for model in MODELS_FOR_DEFAULT_QT
            }
            # Add a fast model that uses the prompts only in the chat history.
            labelers[MODEL_FOR_PROMPT_ONLY_FULL_NAME] = get_quicktake_generator(
                MODEL_FOR_PROMPT_ONLY, chat_history, prompt_only=True, timeout_secs=timeout_secs
            )
            # Add a fine-tuned model that minimizes truncations and formatting in responses.
            labelers[MODEL_FOR_FINETUNE_QT_FULL_NAME] = get_quicktake_generator(
                MODEL_FOR_FINETUNE_QT_FULL_NAME, chat_history, timeout_secs=timeout_secs
            )
            multi_generator = MultiLLMLabeler(
                labelers=labelers,
                timeout_secs=timeout_secs,
                early_terminate_on=MODELS_FOR_DEFAULT_QT,
            )
            max_context_length = min(
                (QT_MAX_CONTEXT_LENGTH.get(model, 16000) for model in MODELS_FOR_DEFAULT_QT), default=16000
            )
            quicktakes = await multi_generator.alabel(
                tiktoken_trim(request.prompt or "", int(max_context_length * 0.75), direction="right")
            )
            quicktake = QT_CANT_ANSWER
            responses_by_model = {model: type(response).__name__ for model, response in quicktakes.items()}
            for model in labelers:
                response = quicktakes.get(model)
                if response and not isinstance(response, Exception):
                    response_model = model
                    quicktake = response
                    break
        elif request.model in get_qt_llms():
            # Specific model requested.
            generator = get_quicktake_generator(request.model, chat_history)
            max_context_length = QT_MAX_CONTEXT_LENGTH.get(request.model, min(QT_MAX_CONTEXT_LENGTH.values()))
            quicktake = await generator.alabel(
                tiktoken_trim(request.prompt or "", int(max_context_length * 0.75), direction="right")
            )
            response_model = request.model
            responses_by_model[request.model] = type(quicktake).__name__
        else:
            raise ValueError(f"Unsupported model: {request.model}; supported: {','.join(get_qt_llms().keys())}")
    except Exception as e:
        log_dict = {
            "message": "Error generating quicktake",
            "model": request.model,
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise e

    end_time = time.time()
    metric_record("quicktake/latency_ms", int((end_time - start_time) * 1000))
    log_dict = {
        "message": f"Quicktake generated with {response_model} in {int((end_time - start_time) * 1000)}ms",
        "is_refusal": str(quicktake == QT_CANT_ANSWER),
        "chat_id": chat_id,
        "turn_id": turn_id,
        "model": response_model,
        "duration_secs": str(end_time - start_time),
        "content_length": str(len(quicktake)),
    }
    for model, response_type in responses_by_model.items():
        log_dict[f"{model}_response_type"] = response_type
    logging.info(json_dumps(log_dict))
    # The client is not aware of these private models, so return its base name; keep the full name in the log above.
    if response_model == MODEL_FOR_PROMPT_ONLY_FULL_NAME:
        response_model = MODEL_FOR_PROMPT_ONLY
    if response_model == MODEL_FOR_FINETUNE_QT_FULL_NAME:
        response_model = MODEL_FOR_FINETUNE_QT
    return QuickTakeResponse(quicktake=quicktake, model=response_model)


async def check_for_stop_request(chat_uuid: UUID, turn_uuid: UUID, model_name: str) -> bool:
    """
    Check if there's a stop request for the current stream.

    Args:
        chat_id: The chat ID
        turn_id: The turn ID
        model_name: The model name

    Returns:
        bool: True if streaming should stop, False otherwise
    """
    try:
        chat_id = str(chat_uuid)
        turn_id = str(turn_uuid)
        redis_client = await get_upstash_redis_client()

        # Create the same keys as in the JS version
        key = f"stop-stream-request-chat:{chat_id}-{turn_id}:model"
        key_without_turn = f"stop-stream-request-chat:{chat_id}-:model"

        # Get multiple values at once
        all_values = await redis_client.mget(key, key_without_turn)

        for value in all_values:
            if value == model_name or value == "":
                logging.info(f"Found stop signal for model: {model_name}, key: {str(key)}")
                # Clear the KV store value since we've detected it
                try:
                    await redis_client.delete(key)
                    await redis_client.delete(key_without_turn)
                except Exception as e:
                    logging.warning(
                        f"Error while attempting to clear stop streaming signal for"
                        f"{chat_id}-{turn_id}-{model_name}: {str(e)}"
                    )
                return True
        return False

    except Exception as e:
        logging.error(f"Error checking stop request: {str(e)}")
        return False


BILLING_ERROR_KEYWORDS = [
    "quota",
    "plan",
    "billing",
    "insufficient",
    "exceeded",
    "limit",
    "payment",
    "subscription",
    "credit",
    "balance",
    "access",
    "unauthorized",
]


def contains_billing_keywords(message: str) -> bool:
    """Checks if a message contains keywords that indicate severe billing/quota issues"""
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in BILLING_ERROR_KEYWORDS)
