import logging
import random
import re
import time
import traceback
from collections.abc import Generator, Mapping, Sequence
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Generic, TypeVar
from uuid import UUID

import pandas as pd
from cachetools.func import ttl_cache
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr
from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import Insert as pg_insert
from sqlalchemy.orm import joinedload
from sqlmodel import Session, select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.config import settings
from ypl.backend.db import get_async_engine, get_engine
from ypl.backend.llm.constants import ACTIVE_MODELS_BY_PROVIDER, PROVIDER_MODEL_PATTERNS, ChatProvider
from ypl.backend.llm.labeler import QT_CANT_ANSWER, MultiLLMLabeler, QuickTakeGenerator
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.prompt_selector import CategorizedPromptModifierSelector, get_modifiers_by_model, store_modifiers
from ypl.backend.llm.provider.provider_clients import get_model_provider_tuple
from ypl.backend.llm.routing.route_data_type import PreferredModel, RoutingPreference
from ypl.backend.llm.utils import GlobalThreadPoolExecutor
from ypl.backend.llm.vendor_langchain_adapter import GeminiLangChainAdapter, OpenAILangChainAdapter
from ypl.backend.utils.json import json_dumps
from ypl.db.attachments import Attachment
from ypl.db.chats import (
    AssistantSelectionSource,
    Chat,
    ChatMessage,
    CompletionStatus,
    MessageType,
    MessageUIStatus,
    PromptModifier,
    PromptModifierAssoc,
    Turn,
)
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, Provider
from ypl.db.redis import get_upstash_redis_client
from ypl.utils import async_timed_cache, tiktoken_trim

RESPONSE_SEPARATOR = " | "
DEFAULT_HIGH_SIM_THRESHOLD = 0.825
DEFAULT_UNIQUENESS_THRESHOLD = 0.75
OPENAI_FT_ID_PATTERN = re.compile(r"^ft:(?P<model>.+?):(?P<organization>.+?)::(?P<id>.+?)$")

YuppMessage = HumanMessage | AIMessage | SystemMessage  # this is needed for proper Pydantic typecasting
YuppMessageRow = list[YuppMessage]


def get_base_model(chat_llm_cls: type[Any] | None, model: str) -> str:
    if chat_llm_cls == ChatOpenAI and (match := OPENAI_FT_ID_PATTERN.match(model)):
        model = match.group("model")

    return model


def get_canonical_model_name(model: str, provider: ChatProvider) -> str:
    """
    Returns the canonical model name for the given provider. Different providers may assign different names to the
    same model, e.g., HuggingFace vs. Google. This function returns the name that is used by the provider.
    """
    match model, provider:
        case "gemma-2-9b-it", ChatProvider.TOGETHER:
            return "google/gemma-2-9b-it"
        case "nemotron-4-340b-instruct", ChatProvider.NVIDIA:
            return "nvidia/nemotron-4-340b-instruct"
        case "yi-large", ChatProvider.NVIDIA:
            return "01-ai/yi-large"
        case "deepseek-coder-v2", ChatProvider.DEEPSEEK:
            return "deepseek-coder"  # this is not a bug. FE does not have v2 in it
        case "qwen1.5-72b-chat", ChatProvider.TOGETHER:
            return "Qwen/Qwen1.5-72B-Chat"
        case "phi-3-mini-4k-instruct", ChatProvider.MICROSOFT:
            return "microsoft/phi-3-mini-4k-instruct"
        case _:
            return model


@lru_cache(maxsize=1024)
def standardize_provider_name(provider: str) -> str:
    return re.sub(r"\s+", "", provider).lower()


def simple_deduce_original_provider(model: str) -> str:
    for pattern, provider in PROVIDER_MODEL_PATTERNS.items():
        if pattern.match(model):
            return provider

    return model


@ttl_cache(ttl=600)  # 10-min cache
def get_all_pro_models() -> Sequence[str]:
    query = select(LanguageModel.internal_name).where(
        LanguageModel.is_pro.is_(True),  # type: ignore
        LanguageModel.deleted_at.is_(None),  # type: ignore
        LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
    )

    with Session(get_engine()) as session:
        return session.exec(query).all()


@ttl_cache(ttl=600)  # 10-min cache
def get_all_strong_models() -> Sequence[str]:
    query = select(LanguageModel.internal_name).where(
        LanguageModel.is_strong.is_(True),  # type: ignore
        LanguageModel.deleted_at.is_(None),  # type: ignore
        LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
    )

    with Session(get_engine()) as session:
        return session.exec(query).all()


@ttl_cache(ttl=600)  # 10-min cache
def get_user_message(turn_id: str) -> str:
    """Returns the user message for the given turn ID. If no user message is found, returns an empty string."""
    query = select(ChatMessage.content).where(
        ChatMessage.turn_id == UUID(turn_id), ChatMessage.message_type == MessageType.USER_MESSAGE
    )

    with Session(get_engine()) as session:
        return session.exec(query).first() or ""


def get_chat_history(
    chat_id: str,
    turn_id: str | None = None,
    merge_method: str = "single-thread-random",  # noqa: C901
    last_num_turns: int = 100,
) -> list[dict[str, Any]]:
    """
    Returns the chat history for the given chat ID.

    Args:
        chat_id: The ID of the chat to get the history for.
        turn_id: The ID of the last turn to include in the history. If None, the history will include all turns.
        merge_method: The method to use to merge the chat history if there are multiple assistant messages. Currently
            only "single-thread-random" is supported, e.g., randomly select one of the assistant messages if none are
            explicitly selected by the user.

    Returns:
        A sequence of {"content": str, "role": str} objects, where role is one of "user", "assistant", or "quicktake".
    """
    query_dict = dict(chat_id=chat_id)

    if turn_id is None:
        sql_query = text(
            f"""
            SELECT cm.message_type, cm.content, t.turn_id, me.score FROM turns t
                JOIN chat_messages cm ON cm.turn_id = t.turn_id
                LEFT JOIN message_evals me ON me.message_id = cm.message_id
            WHERE t.chat_id = :chat_id
            ORDER BY cm.created_at DESC
            LIMIT {last_num_turns}
            """
        )
    else:
        sql_query = text(
            f"""
            WITH turn_seq_id AS (
                SELECT sequence_id FROM turns WHERE turn_id = :turn_id
            )
            SELECT cm.message_type, cm.content, t.turn_id, me.score FROM turns t
                JOIN chat_messages cm ON cm.turn_id = t.turn_id
                LEFT JOIN message_evals me ON me.message_id = cm.message_id
            WHERE t.chat_id = :chat_id
                AND t.sequence_id BETWEEN (SELECT sequence_id FROM turn_seq_id) - {last_num_turns}
                AND (SELECT sequence_id FROM turn_seq_id)
            ORDER BY cm.created_at DESC
            """
        )
        query_dict["turn_id"] = turn_id

    with get_engine().connect() as conn:
        c = conn.execute(sql_query, query_dict)
        rows = c.fetchall()

    last_turn_id = None
    last_turn_id_to_include = turn_id
    history = []
    asst_buffer: list[dict[str, Any]] = []

    for row in reversed(rows):
        if last_turn_id is not None and last_turn_id_to_include == last_turn_id:
            break

        msg_type, content, turn_id, score = row

        if turn_id != last_turn_id:
            if asst_buffer:
                random.shuffle(asst_buffer)
                history.append(
                    dict(
                        content=max(asst_buffer, key=lambda x: x["score"])["content"],
                        role="assistant",
                    )
                )

            asst_buffer.clear()

        last_turn_id = turn_id

        match msg_type:
            case "USER_MESSAGE":
                history.append(dict(content=content, role="user"))
            case "ASSISTANT_MESSAGE":
                asst_buffer.append(dict(content=content, score=0 if score is None else score))
            case "QUICK_RESPONSE_MESSAGE":
                history.append(dict(content=content, role="quicktake"))
            case _:
                continue

    if asst_buffer:
        random.shuffle(asst_buffer)
        history.append(
            dict(
                content=max(asst_buffer, key=lambda x: x["score"])["content"],
                role="assistant",
            )
        )

    return history


def get_preferences(chat_id: str) -> tuple[RoutingPreference, list[str]]:
    """Returns the preferences and user-selected models for the given chat ID."""
    sql_query = text(
        """
        SELECT cm.assistant_model_name, t.turn_id, e.eval_type, me.score, cm.assistant_selection_source FROM turns t
            JOIN chat_messages cm ON cm.turn_id = t.turn_id
            LEFT JOIN message_evals me ON me.message_id = cm.message_id
            LEFT JOIN evals e ON e.eval_id = me.eval_id
        WHERE t.chat_id = :chat_id AND cm.message_type = 'ASSISTANT_MESSAGE'
        ORDER BY cm.created_at DESC
        LIMIT 100
        """
    )

    with get_engine().connect() as conn:
        c = conn.execute(sql_query, dict(chat_id=chat_id))
        rows = c.fetchall()

    df_rows = []

    for row in reversed(rows):  # reverse to get the oldest turn first
        model_name, turn_id, eval_type, score, selection_src = row
        df_rows.append(
            dict(
                model_name=model_name,
                turn_id=turn_id,
                eval_type=eval_type,
                score=score,
                selection_src=selection_src,
            )
        )

    if not df_rows:
        return RoutingPreference(turns=[]), []

    df = pd.DataFrame(df_rows)
    preferred_models_list = []

    for _, gdf in df.groupby("turn_id", sort=False):
        gdf = gdf[gdf["selection_src"] == "ROUTER_SELECTED"]
        evaluated_models = gdf[gdf["eval_type"].isin(["SELECTION", "ALL_BAD"])]["model_name"].tolist()

        if not evaluated_models:
            preferred_models_list.append(
                PreferredModel(models=gdf["model_name"].tolist(), preferred=None, has_evaluation=False)
            )
            continue

        all_models = gdf["model_name"].tolist()
        preferred_models = gdf[(gdf["eval_type"] == "SELECTION") & (gdf["score"] == 100.0)]["model_name"].tolist()
        preferred_model = preferred_models[0] if preferred_models else None
        preferred_models_list.append(PreferredModel(models=all_models, preferred=preferred_model))

    user_selected_models = list(dict.fromkeys(df[df["selection_src"] == "USER_SELECTED"]["model_name"].tolist()))

    return RoutingPreference(turns=preferred_models_list), user_selected_models


def get_shown_models(turn_id: str) -> list[str]:
    """Returns the models shown to the user for the given turn ID."""
    query = (
        select(ChatMessage.assistant_model_name)
        .where(ChatMessage.turn_id == UUID(turn_id), ChatMessage.message_type == MessageType.ASSISTANT_MESSAGE)
        .order_by(ChatMessage.created_at)  # type: ignore[arg-type]
    )

    with Session(get_engine()) as session:
        return session.exec(query).all()  # type: ignore[return-value]


def get_chat(chat_id: str) -> Chat:
    """Returns the chat for the given chat ID."""
    query = select(Chat).where(Chat.chat_id == chat_id)

    with Session(get_engine()) as session:
        chat = session.exec(query).first()

        if chat is None:
            raise ValueError(f"Chat not found for chat_id: {chat_id}")

        return chat


@ttl_cache(ttl=600)  # 10-min cache
def deduce_semantic_groups(models: tuple[str, ...]) -> dict[str, str]:
    """
    Deduces the semantic group of the given model. If a model is not found in the database, its key is absent from the
    returned dictionary.
    """
    semantic_group_map = {}
    sql_query = text(
        """
        SELECT language_models.internal_name, language_models.semantic_group FROM language_models
        WHERE language_models.internal_name IN :model_names
        AND language_models.deleted_at IS NULL
        AND language_models.status = 'ACTIVE'
        AND language_models.semantic_group IS NOT NULL
        """
    )

    with get_engine().connect() as conn:
        c = conn.execute(sql_query, dict(model_names=models))
        rows = c.fetchall()

    for row in rows:
        model, semantic_group = row[0], row[1]
        semantic_group_map[model] = semantic_group.lower().strip()

    return semantic_group_map


@ttl_cache(ttl=600)  # 10-min cache
def deduce_original_providers(models: tuple[str, ...]) -> dict[str, str]:
    provider_map = {}
    models_left = set(models)

    for model in models_left.copy():
        for pattern, provider in PROVIDER_MODEL_PATTERNS.items():
            if pattern.match(model):
                provider_map[model] = provider
                models_left.remove(model)

    if not models_left:
        # We are done
        return provider_map

    models_left_ = tuple(models_left)

    sql_query = text(
        """
        SELECT providers.name, language_models.internal_name FROM language_models
            JOIN providers ON language_models.provider_id = providers.provider_id
        WHERE language_models.internal_name IN :model_names
        """
    )

    with get_engine().connect() as conn:
        c = conn.execute(sql_query, dict(model_names=models_left_))
        rows = c.fetchall()

        for row in rows:
            provider, model = row[0], row[1]
            models_left.remove(model)
            provider_map[model] = standardize_provider_name(provider)

    for model in models_left:
        # On failure, assume model name is provider name
        provider_map[model] = model

    return provider_map


@ttl_cache(ttl=600)  # 10-minute cache
def deduce_original_provider(model: str) -> str:
    for pattern, provider in PROVIDER_MODEL_PATTERNS.items():
        if pattern.match(model):
            return provider

    sql_query = text(
        """
        SELECT providers.name FROM language_models
            JOIN providers ON language_models.provider_id = providers.provider_id
        WHERE language_models.internal_name = :model
        """
    )

    with get_engine().connect() as conn:
        c = conn.execute(sql_query, dict(model=model))
        res = c.first() or (None,)

    name = res[0] or model

    return standardize_provider_name(name)  # return model if all else fails  # type: ignore


@async_timed_cache(seconds=600)
async def adeduce_original_provider(model: str) -> str:
    """Tries to deduce the original provider of a model."""
    # First try to match the model string to a known provider
    for pattern, provider in PROVIDER_MODEL_PATTERNS.items():
        if pattern.match(model):
            return provider

    # If no match was found, try to return the provider in the database
    query = (
        select(Provider.name)
        .join(LanguageModel)
        .where(
            LanguageModel.internal_name == model,
            LanguageModel.deleted_at.is_(None),  # type: ignore
            LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
            Provider.deleted_at.is_(None),  # type: ignore
            Provider.is_active.is_(True),  # type: ignore
        )
        .limit(1)
    )

    async with AsyncSession(get_async_engine()) as session:
        providers = await session.exec(query)

    if provider_ := providers.first():
        return standardize_provider_name(provider_)

    # If all else fails, return the model name
    return model


def get_chat_model(
    info: ModelInfo,
    chat_model_pool: dict[ChatProvider, list[str]] = ACTIVE_MODELS_BY_PROVIDER,
    **chat_kwargs: Any | None,
) -> BaseChatModel:
    """
    Gets the chat model based on the provider and the model name. For OpenAI, Anthropic, Google, Mistral, and HF, the
    only required fields are the provider name, model name, and the API key. For Together, DeepSeek, Nvidia, Anyscale,
    Qwen, and 01, `info` should additionally contain the `base_url` field.
    """
    provider, model, api_key = info.provider, info.model, info.api_key
    chat_kwargs = chat_kwargs.copy()

    if isinstance(provider, str):
        provider = ChatProvider.from_string(provider)

    chat_llms = {
        ChatProvider.OPENAI: ChatOpenAI,
        ChatProvider.ANTHROPIC: ChatAnthropic,
        ChatProvider.GOOGLE: ChatGoogleGenerativeAI,
        ChatProvider.MISTRAL: ChatMistralAI,
        ChatProvider.TOGETHER: ChatOpenAI,
        ChatProvider.DEEPSEEK: ChatOpenAI,
        ChatProvider.NVIDIA: ChatOpenAI,
        ChatProvider.ANYSCALE: ChatOpenAI,
        ChatProvider.ZERO_ONE: ChatOpenAI,
        ChatProvider.QWEN: ChatOpenAI,
        ChatProvider.MICROSOFT: ChatOpenAI,
    }

    chat_llm_cls = chat_llms.get(provider)
    full_model = get_canonical_model_name(model, provider)
    base_model = get_base_model(chat_llm_cls, model)

    if not chat_llm_cls:
        raise ValueError(f"Unsupported provider: {provider}")

    if base_model not in chat_model_pool.get(provider, []):
        raise ValueError(f"Unsupported model: {base_model} for provider: {provider}")

    if info.base_url and "base_url" not in chat_kwargs:
        chat_kwargs["base_url"] = info.base_url

    if full_model.startswith("o1") and provider == ChatProvider.OPENAI:
        chat_kwargs["temperature"] = 1  # temperature must be 1 for o1 models

    return chat_llm_cls(api_key=SecretStr(api_key), model=full_model, **chat_kwargs)  # type: ignore


class SelectIntent(str, Enum):
    NEW_CHAT = "new_chat"
    NEW_TURN = "new_turn"
    SHOW_ME_MORE = "show_me_more"


class SelectModelsV2Request(BaseModel):
    intent: SelectIntent
    prompt: str | None = None  # prompt to use for routing
    num_models: int = 2  # number of models to select
    required_models: list[str] | None = None  # models selected explicitly by the user
    chat_id: str | None = None  # chat ID to use for routing
    turn_id: str | None = None  # turn ID to use for routing
    provided_categories: list[str] | None = None  # categories provided by the user


class SelectModelsV2Response(BaseModel):
    models: list[tuple[str, list[tuple[str, str]]]]  # list of (model, list[(prompt modifier ID, prompt modifier)])
    provider_map: dict[str, str]  # map from model to provider
    fallback_models: list[tuple[str, list[tuple[str, str]]]]  # list of fallback models and modifiers


async def select_models_plus(request: SelectModelsV2Request) -> SelectModelsV2Response:
    from ypl.backend.llm.routing.router import RouterState, get_simple_pro_router

    async def select_models_(
        required_models: list[str] | None = None,
        show_me_more_models: list[str] | None = None,
        provided_categories: list[str] | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        num_models = request.num_models
        router = await get_simple_pro_router(
            prompt,
            num_models,
            preference,
            user_selected_models=required_models,
            show_me_more_models=show_me_more_models,
            provided_categories=provided_categories,
        )
        all_models_state = RouterState.new_all_models_state()
        selected_models = router.select_models(state=all_models_state)
        return_models = selected_models.get_sorted_selected_models()

        all_fallback_models = RouterState.new_all_models_state()
        all_fallback_models = all_fallback_models.emplaced(
            all_models=all_fallback_models.all_models.difference(return_models)
        )
        fallback_models = router.select_models(state=all_fallback_models).get_sorted_selected_models()

        return return_models, fallback_models, selected_models.applicable_modifiers

    match request.intent:
        case SelectIntent.NEW_CHAT | SelectIntent.NEW_TURN:
            assert request.prompt is not None, "prompt is required for NEW_CHAT or NEW_TURN intent"
            prompt = request.prompt
        case SelectIntent.SHOW_ME_MORE:
            assert request.turn_id is not None, "turn_id is required for SHOW_ME_MORE intent"
            prompt = get_user_message(request.turn_id)

    match request.intent:
        case SelectIntent.NEW_TURN:
            preference, user_selected_models = get_preferences(request.chat_id)  # type: ignore[arg-type]
            request.required_models = list(dict.fromkeys((request.required_models or []) + user_selected_models))
        case SelectIntent.SHOW_ME_MORE:
            preference, user_selected_models = get_preferences(request.chat_id)  # type: ignore[arg-type]
            preference.turns = preference.turns or []
            preference.turns.append(PreferredModel(models=user_selected_models, preferred=None))
            request.required_models = []
        case _:
            preference = RoutingPreference(turns=[])

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

    models, fallback_models, applicable_modifiers = await select_models_(
        required_models=models,
        show_me_more_models=show_me_more_models,
        provided_categories=request.provided_categories,
    )

    models = models[: request.num_models]
    fallback_models = fallback_models[: request.num_models]

    try:
        selector = CategorizedPromptModifierSelector.make_default_from_db()

        if request.chat_id:
            modifier_history = get_modifiers_by_model(request.chat_id)
        else:
            modifier_history = {}

        prompt_modifiers = selector.select_modifiers(models + fallback_models, modifier_history, applicable_modifiers)

        if request.turn_id:
            GlobalThreadPoolExecutor.get_instance().submit(store_modifiers, request.turn_id, prompt_modifiers)
    except Exception as e:
        logging.error(f"Error selecting modifiers: {e}")
        prompt_modifiers = {}

    return SelectModelsV2Response(
        models=[(model, prompt_modifiers.get(model, [])) for model in models],
        fallback_models=[(model, prompt_modifiers.get(model, [])) for model in fallback_models],
        provider_map=deduce_original_providers(tuple(models)),
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

    if use_all_models_in_chat_history:
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
            messages.append(AIMessage(content=RESPONSE_SEPARATOR.join(all_content)))
    else:
        # Try to find message with SELECTED status
        selected_msg = next(
            (msg for msg in assistant_msgs if msg.ui_status == MessageUIStatus.SELECTED),
            assistant_msgs[0],  # Fallback to first message if none selected
        )
        # if content is null, a place holder is added as part of sanitize_messages.py/replace_empty_messages()
        messages.append(AIMessage(content=selected_msg.content))

    return messages


async def get_curated_chat_context(
    chat_id: UUID,
    use_all_models_in_chat_history: bool,
    model: str,
    current_turn_seq_num: int | None = None,
) -> list[BaseMessage]:
    """Fetch chat history and format it for OpenAI context.

    Args:
        chat_id: The chat ID to fetch history for.
        use_all_models_in_chat_history: Whether to include all models in the chat history.
        model: The model to fetch history for.
        current_turn_seq_num: The sequence number of the current turn.
    """
    if not current_turn_seq_num:
        return []
    query = (
        select(ChatMessage)
        .join(Turn, Turn.turn_id == ChatMessage.turn_id)  # type: ignore[arg-type]
        .join(Chat, Chat.chat_id == Turn.chat_id)  # type: ignore[arg-type]
        .outerjoin(Attachment, Attachment.chat_message_id == ChatMessage.message_id)  # type: ignore[arg-type]
        .options(joinedload(ChatMessage.assistant_language_model).load_only(LanguageModel.internal_name))  # type: ignore
        .where(
            Chat.chat_id == chat_id,
            ChatMessage.deleted_at.is_(None),  # type: ignore[union-attr]
            Turn.deleted_at.is_(None),  # type: ignore[union-attr]
            Chat.deleted_at.is_(None),  # type: ignore[union-attr]
        )
        .order_by(
            Turn.sequence_id.asc(),  # type: ignore[attr-defined]
            ChatMessage.turn_sequence_number.asc(),  # type: ignore[union-attr]
        )
    )
    if current_turn_seq_num:
        query = query.where(Turn.sequence_id < current_turn_seq_num)

    async with AsyncSession(get_async_engine()) as session:
        result = await session.exec(query)
        messages = result.all()

    # Group messages by turn_id
    turns: dict[UUID, list[ChatMessage]] = {}
    for msg in messages:
        if msg.turn_id not in turns:
            turns[msg.turn_id] = []
        turns[msg.turn_id].append(msg)

    formatted_messages: list[BaseMessage] = []
    for turn_messages in turns.values():
        # Get user messages
        user_msgs = [msg for msg in turn_messages if msg.message_type == MessageType.USER_MESSAGE]
        if user_msgs:
            formatted_messages.append(HumanMessage(content=user_msgs[0].content))

        # Get assistant messages
        formatted_messages.extend(_get_assistant_messages(turn_messages, model, use_all_models_in_chat_history))

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


GPT_4O_MINI_LLM = None
GPT_4O_LLM = None
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
            "gemini-1.5-flash-002": get_gemini_15_flash_llm(),
            "gemini-2.0-flash-exp": get_gemini_2_flash_llm(),
        }
    return QT_LLMS


# Models to use if no specific model was requested.
MODELS_FOR_DEFAULT_QT = ["gpt-4o", "gpt-4o-mini", "gemini-2.0-flash-exp"]
# Model to use while supplying only the prompts from the chat history, instead of the full chat history.
MODEL_FOR_PROMPT_ONLY = "gpt-4o"
MODEL_FOR_PROMPT_ONLY_FULL_NAME = MODEL_FOR_PROMPT_ONLY + ":prompt-only"


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
    log_dict = {
        "message": "Quicktake generated",
        "model": response_model,
        "duration_secs": str(end_time - start_time),
        "content_length": str(len(quicktake)),
    }
    logging.info(json_dumps(log_dict))
    # The client is not aware of this private model, so return its base name; keep the full name in the log above.
    if response_model == MODEL_FOR_PROMPT_ONLY_FULL_NAME:
        response_model = MODEL_FOR_PROMPT_ONLY
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
                logging.info(f"found stop signal for mode:{model_name}, key:{str(value)}")
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
