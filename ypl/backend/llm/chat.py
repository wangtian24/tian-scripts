import asyncio
import logging
import time
import traceback
import uuid
from collections import defaultdict
from collections.abc import Generator, Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar
from uuid import UUID

from cachetools.func import ttl_cache
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import Insert as pg_insert
from sqlalchemy.orm import joinedload
from sqlmodel import Session, or_, select, update
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.config import settings
from ypl.backend.db import get_async_engine, get_async_session, get_engine
from ypl.backend.llm.attachment import get_attachments
from ypl.backend.llm.constants import ACTIVE_MODELS_BY_PROVIDER, ChatProvider
from ypl.backend.llm.db_helpers import (
    deduce_original_providers,
    get_chat_model,
    get_chat_required_models,
    get_model_context_lengths,
    get_preferences,
    get_user_message,
    notnull,
)
from ypl.backend.llm.labeler import QT_CANT_ANSWER, QuickTakeGenerator
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.model_heuristics import ModelHeuristics
from ypl.backend.llm.prompt_modifier import run_prompt_modifier_on_models
from ypl.backend.llm.prompt_selector import (
    CategorizedPromptModifierSelector,
    get_modifiers_by_model_and_position,
    store_modifiers,
)
from ypl.backend.llm.provider.provider_clients import get_model_provider_tuple
from ypl.backend.llm.routing.debug import RoutingDebugInfo, build_routing_debug_info
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.routing.router_state import RouterState
from ypl.backend.llm.transform_messages import TransformOptions, transform_user_messages
from ypl.backend.llm.turn_quality import label_turn_quality
from ypl.backend.llm.vendor_langchain_adapter import GeminiLangChainAdapter, OpenAILangChainAdapter
from ypl.backend.prompts import (
    ALL_MODELS_IN_CHAT_HISTORY_PREAMBLE,
    RESPONSE_SEPARATOR,
    SYSTEM_QUICKTAKE_FALLBACK_PROMPT,
    SYSTEM_QUICKTAKE_PROMPT,
    USER_QUICKTAKE_FALLBACK_PROMPT,
    USER_QUICKTAKE_PROMPT,
)
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.monitoring import metric_inc, metric_inc_by, metric_record
from ypl.backend.utils.utils import StopWatch
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
from ypl.utils import Delegator, get_text_part, replace_text_part, tiktoken_trim

MAX_LOGGED_MESSAGE_LENGTH = 200
DEFAULT_HIGH_SIM_THRESHOLD = 0.825
DEFAULT_UNIQUENESS_THRESHOLD = 0.75
YuppMessage = HumanMessage | AIMessage | SystemMessage  # this is needed for proper Pydantic typecasting
YuppMessageRow = list[YuppMessage]
IMAGE_CATEGORY = "image"
IMAGE_ATTACHMENT_MIME_TYPE_SQL_PATTERN = "image/%"


class SelectIntent(str, Enum):
    NEW_CHAT = "new_chat"
    NEW_TURN = "new_turn"
    # TODO(bhanu): remove this after consolidating enum in YuppHead
    SHOW_ME_MORE = "show_me_more"
    SHOW_MORE_WITH_SAME_TURN = "show_more_with_same_turn"
    RETRY = "retry"
    NEW_STYLE = "new_style"


class SelectModelsV2Request(BaseModel):
    user_id: str | None = None
    intent: SelectIntent
    prompt: str | None = None  # prompt to use for routing
    num_models: int = 2  # number of models to select
    required_models: list[str] | None = None  # models selected explicitly by the user
    chat_id: str  # chat ID to use for routing
    turn_id: str  # turn ID to use for routing
    provided_categories: list[str] | None = None  # categories provided by the user
    debug_level: int = 0  # 0: return no debug info, log only, 1: return debug
    prompt_modifier_id: uuid.UUID | None = None  # prompt modifier ID to apply to all models
    # whether to set prompt modifiers automatically, if prompt_modifier_id is not provided
    auto_select_prompt_modifiers: bool = False


class SelectedModelType(str, Enum):
    PRIMARY = "primary"
    FALLBACK = "fallback"


class SelectedModelInfo(BaseModel):
    model: str  # the model name
    provider: str  # the provider name
    type: SelectedModelType
    prompt_modfiers: list[tuple[str, str]]  # a list of (prompt modifier ID, prompt modifier)


class SelectModelsV2Response(BaseModel):
    # legacy model information
    models: list[tuple[str, list[tuple[str, str]]]]  # list of (model, list[(prompt modifier ID, prompt modifier)])
    provider_map: dict[str, str]  # map from model to provider
    fallback_models: list[tuple[str, list[tuple[str, str]]]]  # list of fallback models and modifiers

    # new version of model information, added on 2/1/25, we shall migrate to this gradually
    selected_models: list[SelectedModelInfo]

    routing_debug_info: RoutingDebugInfo | None = None
    num_models_remaining: int | None = None


async def has_image_attachments(chat_id: str) -> bool:
    async with get_async_session() as session:
        result = await session.exec(
            select(ChatMessage.message_id)
            .join(Attachment)
            .join(Turn)
            .where(
                Turn.chat_id == UUID(chat_id),
                ChatMessage.message_type == MessageType.USER_MESSAGE,
                ChatMessage.message_id == Attachment.chat_message_id,
                Attachment.content_type.like(IMAGE_ATTACHMENT_MIME_TYPE_SQL_PATTERN),  # type: ignore
            )
            .limit(1)
        )
        return result.first() is not None


def _set_prompt_modifiers(
    request: SelectModelsV2Request,
    selected_models_rs: RouterState,
) -> dict[str, list[tuple[str, str]]]:
    """
    Sets prompt modifiers for the selected models.

    Logic:
    - If modifiers were applied to the last two messages, use them to set the modifier for the corresponding messages
      in terms of the position of the model in the selection (LHS or RHS).
    - If prompt_modifier_id is provided, use it to set the prompt modifier for unmodified models.
    - If a model has previously been modified in a certain way, use the same modifier for it.
    - If a model is unmodified after all checks above, and `auto_select_prompt_modifiers` is set in the request,
      select a random modifier for it.
    """
    prompt_modifiers: dict[str, list[tuple[str, str]]] = {}

    selected_models = selected_models_rs.get_sorted_selected_models()

    modifier_selector = CategorizedPromptModifierSelector.make_default_from_db()
    if request.intent == SelectIntent.NEW_CHAT:
        if request.prompt_modifier_id:
            modifier = modifier_selector.modifiers_by_id.get(str(request.prompt_modifier_id))
            if modifier:
                prompt_modifiers = {m: [(str(request.prompt_modifier_id), modifier.text)] for m in (selected_models)}
            else:
                logging.warning(f"Ignoring unknown modifier ID: {request.prompt_modifier_id}")
        else:
            return prompt_modifiers

    else:
        try:
            if request.chat_id and request.intent != SelectIntent.NEW_CHAT:
                modifier_history, modifiers_by_position = get_modifiers_by_model_and_position(request.chat_id)
            else:
                modifier_history, modifiers_by_position = {}, (None, None)

            should_auto_select_modifiers = request.auto_select_prompt_modifiers and request.intent not in (
                SelectIntent.SHOW_ME_MORE,
                SelectIntent.SHOW_MORE_WITH_SAME_TURN,
            )

            prompt_modifiers = modifier_selector.select_modifiers(
                selected_models,
                modifier_history,
                selected_models_rs.applicable_modifiers,
                user_selected_modifier_id=request.prompt_modifier_id,
                modifiers_by_position=modifiers_by_position,
                should_auto_select_modifiers=should_auto_select_modifiers,
            )
        except Exception as e:
            logging.error(f"Error selecting modifiers: {e}")

    if request.turn_id and prompt_modifiers:
        asyncio.create_task(store_modifiers(request.turn_id, prompt_modifiers))

    return prompt_modifiers


def kick_off_label_turn_quality(prompt: str, chat_id: str, turn_id: str) -> str:
    assert prompt is not None, "prompt is required for NEW_CHAT or NEW_TURN intent"
    asyncio.create_task(label_turn_quality(UUID(turn_id), UUID(chat_id), prompt))
    return prompt


async def select_models_plus(request: SelectModelsV2Request) -> SelectModelsV2Response:
    from ypl.backend.llm.routing.router import get_simple_pro_router

    logging.debug(json_dumps({"message": "select_models_plus request"} | request.model_dump(mode="json")))
    metric_inc(f"routing/intent_{request.intent}")
    stopwatch = StopWatch()

    prompt = None
    match request.intent:
        case SelectIntent.NEW_CHAT:
            prompt = kick_off_label_turn_quality(notnull(request.prompt), request.chat_id, request.turn_id)
            preference = RoutingPreference(
                turns=[],
                user_selected_models=request.required_models or [],  # get from the request
                same_turn_shown_models=[],
                user_id=request.user_id,
            )
        case SelectIntent.NEW_TURN:
            prompt = kick_off_label_turn_quality(notnull(request.prompt), request.chat_id, request.turn_id)
            preference = get_preferences(request.user_id, request.chat_id, request.turn_id)
        case SelectIntent.SHOW_ME_MORE:
            prompt = get_user_message(request.turn_id)
            preference = get_preferences(request.user_id, request.chat_id, request.turn_id)
        case _:
            raise ValueError(f"Unsupported intent {request.intent} for select_models_plus()")

    stopwatch.record_split("get_preference")

    preference.debug_level = request.debug_level

    # Figure out what models are required for routing (must appear)
    # 1. get it from request, if none, from DB, if none, from preference (inferred from history messages in the chat)
    # 2. add all previous turn non-downvoted models for stability (2/4/2025, see routing decision log)
    # 3. remove all same-turn already-shown user-selected models (when it's SMM)
    required_models = request.required_models
    if not required_models:
        required_models = list(get_chat_required_models(UUID(request.chat_id)))
    if not required_models:
        required_models = preference.user_selected_models or []

    required_models_for_routing = list(required_models) + preference.get_inherited_models(
        is_show_me_more=(request.intent == SelectIntent.SHOW_ME_MORE)
    )
    required_models_for_routing = (
        [m for m in required_models_for_routing if m not in (preference.same_turn_shown_models or [])]
        if request.intent == SelectIntent.SHOW_ME_MORE
        else required_models_for_routing
    )
    required_models_for_routing = list(dict.fromkeys(required_models_for_routing))  # just dedupe

    stopwatch.record_split("infer_required_models")

    # only run the routing chain if we don't have enough models to serve.
    primary_models = []
    primary_models_rs = None
    if len(required_models_for_routing) >= request.num_models:
        primary_models = required_models_for_routing[: request.num_models]
        num_models_remaining = request.num_models
    else:
        # create a router
        router = await get_simple_pro_router(
            prompt,
            request.num_models,
            preference,
            required_models=required_models_for_routing,
            show_me_more_models=preference.same_turn_shown_models or [],
            provided_categories=request.provided_categories or [],
            chat_id=request.chat_id,
            is_new_turn=(request.intent == SelectIntent.NEW_TURN),
            for_fallback=True,
        )
        # actually run the router chain
        all_models_state = RouterState.new_all_models_state()
        primary_models_rs = router.select_models(state=all_models_state)
        # extract results
        selected_models = primary_models_rs.get_selected_models()
        primary_models = selected_models[: request.num_models]
        fallback_models = selected_models[request.num_models : request.num_models * 2]
        num_models_remaining = primary_models_rs.num_models_remaining or request.num_models

    stopwatch.record_split("routing_primary")

    # Update the required models, remove models already chosen in the primary routing
    required_models_for_routing = [m for m in required_models_for_routing if m not in primary_models]

    fallback_models = []
    fallback_models_rs = None
    if len(required_models_for_routing) >= request.num_models:
        fallback_models = required_models_for_routing[request.num_models :]
    else:
        router = await get_simple_pro_router(
            prompt,
            request.num_models,
            preference,
            # exclude the user selected models already used in the primary selection
            required_models=required_models_for_routing,
            show_me_more_models=preference.same_turn_shown_models or [],
            provided_categories=request.provided_categories or [],
            is_new_turn=(request.intent == SelectIntent.NEW_TURN),
            for_fallback=True,
        )
        all_fallback_models = RouterState.new_all_models_state()
        all_fallback_models = all_fallback_models.emplaced(
            # just keep the models not already in the return models (any remainders), select within them
            all_models=all_fallback_models.all_models.difference(primary_models)
        )
        fallback_models_rs = router.select_models(state=all_fallback_models)
        fallback_models = fallback_models_rs.get_selected_models()

    stopwatch.record_split("routing_fallback")

    # Generate modifier labels for the models we selected
    prompt_modifiers_rs = await run_prompt_modifier_on_models(prompt, primary_models + fallback_models)
    prompt_modifiers = _set_prompt_modifiers(request, prompt_modifiers_rs)

    stopwatch.record_split("prompt_modifiers")

    routing_debug_info: RoutingDebugInfo = build_routing_debug_info(
        primary_models=primary_models,
        fallback_models=fallback_models,
        router_state=primary_models_rs,
        required_models=request.required_models,
    )

    # Debug logging
    if logging.root.getEffectiveLevel() == logging.DEBUG:
        for model, model_debug in routing_debug_info.model_debug.items():
            is_selected = " S => " if model in primary_models else ""
            logging.debug(f"> {model:<50}{is_selected:<8}{model_debug.score:-10.1f}{model_debug.journey}")

    # if the number of returned models is different from what the requests asked, log more information
    if len(primary_models) != request.num_models:
        log_dict = {
            "message": f"Model routing: requested {request.num_models} models, but returned {len(primary_models)}.",
            "chat_id": request.chat_id,
            "turn_id": request.turn_id,
            "intent": request.intent,
        }
        logging.info(json_dumps(log_dict))

    providers_by_model = deduce_original_providers(tuple(primary_models + fallback_models))

    # metric counters and logging
    metric_inc_by("routing/count_models_served", len(primary_models))
    if len(primary_models) > 0:
        metric_inc(f"routing/count_first_{primary_models[0]}")
    if len(primary_models) > 1:
        metric_inc(f"routing/count_second_{primary_models[1]}")
    for model in primary_models:
        metric_inc(f"routing/count_chosen_{model}")
    stopwatch.record_split("prepare_response")
    stopwatch.export_metrics("routing/latency/")

    response = SelectModelsV2Response(
        models=[(model, prompt_modifiers.get(model, [])) for model in primary_models],
        fallback_models=[(model, prompt_modifiers.get(model, [])) for model in fallback_models],
        provider_map=providers_by_model,
        # another convenient version
        selected_models=[
            SelectedModelInfo(
                model=model,
                provider=providers_by_model[model],
                type=SelectedModelType.PRIMARY if model in primary_models else SelectedModelType.FALLBACK,
                prompt_modfiers=prompt_modifiers.get(model, []),
            )
            for model in primary_models + fallback_models
        ],
        routing_debug_info=routing_debug_info if request.debug_level > 0 else None,
        num_models_remaining=num_models_remaining,
    )
    logging.debug(json_dumps({"message": "select_models_plus response"} | response.model_dump(mode="json")))

    return response


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
    max_message_length: int | None = None,
) -> list[BaseMessage]:
    """Get assistant messages for a turn.

    If use_all_models_in_chat_history is True, includes assistant messages from all models, indicating which ones
    are from the current model and which one was preferred by the user (if any).
    If use_all_models_in_chat_history is False, includes only the preferred messages, or the first message if none
    were selected.
    """
    messages: list[BaseMessage] = []
    assistant_msgs = [msg for msg in turn_messages if msg.message_type == MessageType.ASSISTANT_MESSAGE and msg.content]
    if not assistant_msgs:
        return messages

    if use_all_models_in_chat_history and len(assistant_msgs) > 1:
        all_content = []
        for msg in assistant_msgs:
            content = msg.content or ""
            if max_message_length and len(content) > max_message_length:
                content = content[:max_message_length] + "..."
            if msg.assistant_language_model.internal_name == model:
                # A previous response from the current assistant.
                if content:
                    content = "This was your response:\n\n" + content
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
            content = ""
            if model:
                content += ALL_MODELS_IN_CHAT_HISTORY_PREAMBLE
            content += RESPONSE_SEPARATOR.join(all_content)
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


def _get_enhanced_user_message(messages: list[ChatMessage], max_message_length: int | None = None) -> HumanMessage:
    user_msgs = [msg for msg in messages if msg.message_type == MessageType.USER_MESSAGE]
    if not user_msgs:
        raise ValueError("No user messages found")
    if len(user_msgs) > 1:
        raise ValueError("Multiple user messages found")
    user_msg = user_msgs[0]
    attachments = user_msg.attachments or []
    content = (
        user_msg.content[:max_message_length] + "..."
        if max_message_length and len(user_msg.content) > max_message_length
        else user_msg.content
    )
    return HumanMessage(
        content=content,
        additional_kwargs={"attachments": attachments},
    )


@ttl_cache(ttl=10)  # Don't cache for long - chat history can change. This is for rapid subsequent lookups.
async def get_curated_chat_context(
    chat_id: UUID,
    use_all_models_in_chat_history: bool,
    model: str,
    current_turn_id: UUID | None = None,
    include_current_turn: bool = False,
    max_turns: int = 20,
    max_message_length: int | None = None,
    context_for_logging: str | None = None,
) -> list[BaseMessage]:
    """Fetch chat history and format it for OpenAI context.

    Args:
        chat_id: The chat ID to fetch history for.
        use_all_models_in_chat_history: Whether to include all models in the chat history.
        model: The model to fetch history for.
        current_turn_id: The current turn ID.
        include_current_turn: Whether to include the current turn in the chat history.
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
            or_(
                # Do not include errored responses.
                ChatMessage.completion_status == CompletionStatus.SUCCESS,
                ChatMessage.completion_status.is_(None),  # type: ignore[attr-defined]
            ),
            Chat.deleted_at.is_(None),  # type: ignore[union-attr]
            Turn.turn_id.in_(  # type: ignore[attr-defined]
                select(Turn.turn_id)
                .where(Turn.chat_id == chat_id)
                .order_by(Turn.sequence_id.desc())  # type: ignore[attr-defined]
                .limit(max_turns)
            ),
        )
        .order_by(
            Turn.sequence_id.asc(),  # type: ignore[attr-defined]
            ChatMessage.turn_sequence_number.asc(),  # type: ignore[union-attr]
        )
    )
    if not include_current_turn:
        query = query.where(Turn.turn_id != current_turn_id)

    formatted_messages: list[BaseMessage] = []
    # An async session is 2-3X slower.
    with Session(get_engine()) as session:
        result = session.exec(query)
        # Limit to the most recent messages.
        messages = result.unique().all()

    # Group messages by turn_id
    turns: defaultdict[UUID, list[ChatMessage]] = defaultdict(list)
    for msg in messages:
        turns[msg.turn_id].append(msg)

    # The loop below proceeds in insertion order, which is critical for
    # the correctness of this method.
    for turn_messages in turns.values():
        # Get user messages
        formatted_messages.append(_get_enhanced_user_message(turn_messages, max_message_length))
        # Get assistant messages
        formatted_messages.extend(
            _get_assistant_messages(turn_messages, model, use_all_models_in_chat_history, max_message_length)
        )

    info = {
        "message": f"chat_context ({context_for_logging or 'no context'})",
        "chat_id": str(chat_id),
        "model": model,
    }
    for i, fmsg in enumerate(formatted_messages):
        msg_type = (
            "Human"
            if isinstance(fmsg, HumanMessage)
            else "AI"
            if isinstance(fmsg, AIMessage)
            else "Sys"
            if isinstance(fmsg, SystemMessage)
            else type(fmsg).__name__
        )
        info[f"m{i}_{msg_type}"] = (
            str(fmsg.content[:MAX_LOGGED_MESSAGE_LENGTH]) + "..."
            if len(fmsg.content) > MAX_LOGGED_MESSAGE_LENGTH
            else str(fmsg.content)
        )
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

# For fallback.
MODELS_FOR_FALLBACK = ["gemini-1.5-flash-002"]  # can add others later

# Attachment support
QT_MODEL_WITH_PDF_SUPPORT = ["gemini-2.0-flash-exp"]
QT_MODEL_WITH_IMAGE_SUPPORT = ["gpt-4o", "gpt-4o-mini", "gemini-2.0-flash-exp"]


GPT_4O_MINI_LLM: OpenAILangChainAdapter | None = None
GPT_4O_LLM: OpenAILangChainAdapter | None = None
FINE_TUNED_GPT_4O_LLM: OpenAILangChainAdapter | None = None
GEMINI_15_FLASH_LLM: GeminiLangChainAdapter | None = None
GEMINI_2_FLASH_LLM: GeminiLangChainAdapter | None = None


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
                max_output_tokens=64,
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


QT_LLMS: dict[str, BaseChatModel] | None = None
DEFAULT_QT_MAX_CONTEXT_LENGTH = 128000  # gpt-4o-mini


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
    errors: str | None = None


class QuickTakeRequest(BaseModel):
    user_id: str | None = None
    chat_id: str | None = None
    turn_id: str | None = None
    prompt: str | None = None
    attachment_ids: list[UUID] | None = None
    model: str | None = None  # one of the entries in QT_LLMS; if none, use MODELS_FOR_DEFAULT_QT
    timeout_secs: float | None = None


class PromptModifierInfo(BaseModel):
    prompt_modifier_id: str
    name: str
    description: str | None = None


def create_quicktake_generator(
    model: str,
    chat_history: list[BaseMessage],
    prompt_only: bool = False,
    timeout_secs: float = settings.DEFAULT_QT_TIMEOUT_SECS,
    user_prompt: str = USER_QUICKTAKE_PROMPT,
    system_prompt: str = SYSTEM_QUICKTAKE_PROMPT,
) -> QuickTakeGenerator:
    """Get a quicktake generator for a given model, or raise if the model is not supported."""
    if prompt_only:
        # Use only the prompts from the chat history.
        chat_history = [m for m in chat_history if isinstance(m, HumanMessage)]
    return QuickTakeGenerator(
        get_qt_llms()[model],
        chat_history,
        model_name=model,
        timeout_secs=timeout_secs,
        user_quicktake_prompt=user_prompt,
        system_quicktake_prompt=system_prompt,
        on_error="raise",
    )


async def generate_quicktake(
    request: QuickTakeRequest,
    chat_history: list[BaseMessage] | None = None,
) -> QuickTakeResponse:
    """
    Generates a quicktake for a given chat_id or chat_history. If chat_history is provided, it will be used instead of
    chat_id and turn_id.

    Args:
        chat_id: The chat ID to fetch history for.
        turn_id: The turn ID to fetch history for.
        chat_history: The chat history to use.
    """
    start_time = time.time()

    match request.chat_id, request.turn_id, chat_history:
        case None, None, None:
            raise ValueError("Either chat_id or chat_history must be provided")
        case None, None, _:
            pass
        case _, _, None:
            assert request.chat_id is not None  # because mypy cannot infer this
            turn_id = uuid.UUID(request.turn_id) if request.turn_id else None
            chat_history = await get_curated_chat_context(
                chat_id=uuid.UUID(request.chat_id),
                use_all_models_in_chat_history=False,
                model=request.model or "",
                current_turn_id=turn_id,
                context_for_logging="quicktake",
            )

    assert chat_history is not None, "chat_history is null"

    chat_history_time = time.time() - start_time

    # Add attachments (image, etc) to the chat history, as a url or base64 encoded string.
    old_attachments = [attachment for m in chat_history for attachment in m.additional_kwargs.get("attachments", [])]
    new_attachments = await get_attachments(request.attachment_ids) if request.attachment_ids else []
    all_attachments = old_attachments + new_attachments
    has_attachments = len(all_attachments) > 0
    has_pdf_attachments = any(attachment.content_type == "application/pdf" for attachment in all_attachments)
    has_image_attachments = any(
        attachment.content_type is not None and attachment.content_type.startswith("image/")
        for attachment in all_attachments
    )
    transform_options: TransformOptions = {"image_type": "thumbnail", "use_signed_url": False}
    chat_history = await transform_user_messages(chat_history, QT_MODEL_WITH_PDF_SUPPORT[0], options=transform_options)

    # Calculate the length of input with all the information, and check against the context length allowed by the model.
    chat_history_text = "\n".join(get_text_part(m) for m in chat_history)
    chat_history_context_len = len(ModelHeuristics(tokenizer_type="tiktoken").encode_tokens(chat_history_text))
    min_required_context_len = int(chat_history_context_len * 1.2)  # Add a buffer for system prompt etc.
    context_lengths = get_model_context_lengths()

    # Choose models to use for generating quicktakes, we have a set of main models that try to answer the question,
    # and a set of fallback models that are fast but only try to provide contextual commentaries.
    # TODO(tian): no need to prefilter the models as we are trimming the message to fit their context length anyway.
    qt_models = [
        model
        for model in MODELS_FOR_DEFAULT_QT
        if context_lengths.get(model, DEFAULT_QT_MAX_CONTEXT_LENGTH) > min_required_context_len
    ]
    fallback_models = [
        model
        for model in MODELS_FOR_FALLBACK
        if context_lengths.get(model, DEFAULT_QT_MAX_CONTEXT_LENGTH) > min_required_context_len
    ]
    _model_max_context_lengths = {k: v for k, v in context_lengths.items() if k in qt_models + fallback_models}

    timeout_secs = (
        request.timeout_secs  # the timeout provided by the client will override the defaults
        if request.timeout_secs
        else settings.ATTACHMENT_QUICKTAKE_TIMEOUT_SECS
        if has_attachments
        else settings.DEFAULT_QT_TIMEOUT_SECS
    )

    preferred_models = []
    if request.model:
        preferred_models = [request.model]
    elif has_pdf_attachments or has_image_attachments:
        if has_pdf_attachments:
            preferred_models = QT_MODEL_WITH_PDF_SUPPORT
        elif has_image_attachments:
            preferred_models = QT_MODEL_WITH_IMAGE_SUPPORT
        if not preferred_models:
            logging.warning("No preferred models found, using all models")

    # Transform the latest message with attachments, if any.
    # Supply this to the labelers after trimming the context.
    latest_message_transform_result = await transform_user_messages(
        [
            HumanMessage(
                content=request.prompt or "",
                additional_kwargs={
                    "attachments": new_attachments,
                },
            )
        ],
        QT_MODEL_WITH_PDF_SUPPORT[0],
        options=transform_options,
    )
    _latest_message = HumanMessage(content=latest_message_transform_result[0].content)
    errors = ""

    try:
        # -- Prepare labelers for the main quicktake call
        primary_models = []  # we only early terminate on these models
        if not preferred_models:
            primary_models.extend(qt_models)
        elif preferred_models and all(model in get_qt_llms() for model in preferred_models):
            primary_models.extend(preferred_models)
        else:
            raise ValueError(f"Unsupported model: {preferred_models}; supported: {','.join(get_qt_llms().keys())}")

        secondary_models = (
            [MODEL_FOR_PROMPT_ONLY_FULL_NAME, MODEL_FOR_FINETUNE_QT_FULL_NAME] if not has_attachments else []
        )
        fallback_models = fallback_models if not has_attachments else []

        # We have three tiers of QT models (labelers)
        # 1. primary - high quality but not the fastets, might have refusal as well. Can early terminate.
        # 2. secondary - faster but might not be as good.
        # 3. fallback - fastest but may not fully answer the question, just contextual commentaries.
        all_labelers: dict[str, Any] = {
            # primary labelers
            model: create_quicktake_generator(
                model,
                chat_history,
                user_prompt=USER_QUICKTAKE_PROMPT,
                system_prompt=SYSTEM_QUICKTAKE_PROMPT,
            )
            for model in primary_models
        }

        if secondary_models:
            all_labelers = all_labelers | {
                # secondary labelers
                MODEL_FOR_PROMPT_ONLY_FULL_NAME: create_quicktake_generator(
                    MODEL_FOR_PROMPT_ONLY,
                    chat_history,
                    prompt_only=True,
                    user_prompt=USER_QUICKTAKE_PROMPT,
                    system_prompt=SYSTEM_QUICKTAKE_PROMPT,
                ),
                MODEL_FOR_FINETUNE_QT_FULL_NAME: create_quicktake_generator(
                    MODEL_FOR_FINETUNE_QT_FULL_NAME,
                    chat_history,
                    user_prompt=USER_QUICKTAKE_PROMPT,
                    system_prompt=SYSTEM_QUICKTAKE_PROMPT,
                ),
            }

        if fallback_models:
            all_labelers = all_labelers | {
                # fallback labelers
                model: create_quicktake_generator(
                    model,
                    chat_history,
                    user_prompt=USER_QUICKTAKE_FALLBACK_PROMPT,
                    system_prompt=SYSTEM_QUICKTAKE_FALLBACK_PROMPT,
                )
                for model in fallback_models
            }

        priority_groups = [primary_models, secondary_models, fallback_models]
        # TODO(Raghu): This includes secondary and fallback models even when there is preferred model
        #              or PDF attachment. Decide if we want to include these.

        # -- Make all quicktake calls in parallel
        class LabelerTask:
            model: str
            labeler: QuickTakeGenerator

            def __init__(self, model: str, labeler: QuickTakeGenerator):
                self.model = model
                self.labeler = labeler

            async def async_run(self) -> Any:
                max_context_length = min(
                    _model_max_context_lengths.get(self.model, DEFAULT_QT_MAX_CONTEXT_LENGTH),
                    DEFAULT_QT_MAX_CONTEXT_LENGTH,
                )
                trimmed_message = replace_text_part(
                    _latest_message,
                    self.labeler.user_quicktake_prompt.format(
                        prompt=tiktoken_trim(request.prompt or "", int(max_context_length * 0.75), direction="right")
                    ),
                )
                return await self.labeler.alabel(trimmed_message)

        labeler_tasks = {m: LabelerTask(m, labeler) for m, labeler in all_labelers.items()}
        all_quicktakes: dict[str, Any] = await Delegator(
            labeler_tasks, timeout_secs=timeout_secs, priority_groups=priority_groups
        ).async_run()

        # -- Post-processing
        response_quicktake = QT_CANT_ANSWER
        response_model = ""
        found_response = False
        for model in all_labelers.keys():
            response = all_quicktakes.get(model)
            if response and not isinstance(response, Exception):
                response_model = model
                response_quicktake = response
                found_response = True
                break
        if not found_response:
            errors = "no_response"

    except Exception as e:
        err_log_dict = {
            "message": "Error generating quicktake",
            "model": preferred_models,
            "error": str(e),
        }
        logging.exception(json_dumps(err_log_dict))
        raise e

    # The client is not aware of these private models, so return its base name; keep the full name in the log above.
    # TODO(tian): this is no longer important, remove this later.
    if response_model == MODEL_FOR_PROMPT_ONLY_FULL_NAME:
        response_model = MODEL_FOR_PROMPT_ONLY
    if response_model == MODEL_FOR_FINETUNE_QT_FULL_NAME:
        response_model = MODEL_FOR_FINETUNE_QT

    # Logging and bookkeeping
    end_time = time.time()
    metric_record("quicktake/latency_ms", int((end_time - start_time) * 1000))
    time_taken_ms = int((end_time - start_time) * 1000)
    log_dict: dict[str, Any] = {
        "message": f"Quicktake generated with {response_model} in {time_taken_ms}ms: {response_quicktake}",
        "chat_history": {
            "time_ms": int(chat_history_time * 1000),
            "num_messages": len(chat_history),
            "context_length": chat_history_context_len,
            "text_length": len(chat_history_text),
        },
        # TODO(Raghu): We treat it as refusal even when all the models timeout. Might treat it differently.
        "is_refusal": str(response_quicktake.strip() == QT_CANT_ANSWER),
        "chat_id": request.chat_id,
        "turn_id": request.turn_id,
        "model": response_model,
        "model_responses": [f"{model} -> {str(all_quicktakes[model])}" for model in all_labelers.keys()],
        "duration_secs": end_time - start_time,
        "content_length": len(response_quicktake),
        "old_attachments_ids": [attachment.attachment_id for attachment in old_attachments],
        "new_attachments_ids": [attachment.attachment_id for attachment in new_attachments],
        "attachment_mime_types": [attachment.content_type for attachment in all_attachments],
    }
    logging.info(json_dumps(log_dict))

    return QuickTakeResponse(quicktake=response_quicktake, model=response_model, errors=errors)


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


async def update_failed_message_status(message_id: UUID) -> None:
    """Update completion status of failed message to streaming_error_with_retry"""
    async with get_async_session() as session:
        try:
            query = (
                update(ChatMessage)
                .where(ChatMessage.message_id == message_id)  # type: ignore
                .values(completion_status=CompletionStatus.STREAMING_ERROR_WITH_FALLBACK)
            )
            await session.exec(query)  # type: ignore[call-overload]
            await session.commit()
        except Exception as e:
            info = {
                "message": "Error updating failed message status",
                "error_details": str(e),
            }
            logging.error(json_dumps(info))
