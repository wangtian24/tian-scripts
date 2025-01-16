import logging
import random
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any
from uuid import UUID

import pandas as pd
from cachetools.func import ttl_cache
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from sqlalchemy import text
from sqlmodel import Session, select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.db import get_async_engine, get_engine
from ypl.backend.llm.constants import ACTIVE_MODELS_BY_PROVIDER, PROVIDER_MODEL_PATTERNS, ChatProvider
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.routing.route_data_type import PreferredModel, RoutingPreference
from ypl.db.chats import Chat, ChatMessage, MessageType
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, Provider
from ypl.utils import async_timed_cache

IMAGE_ATTACHMENT_MIME_TYPE = "image/*"


def simple_deduce_original_provider(model: str) -> str:
    for pattern, provider in PROVIDER_MODEL_PATTERNS.items():
        if pattern.match(model):
            return provider

    return model


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


@lru_cache(maxsize=1024)
def standardize_provider_name(provider: str) -> str:
    if provider == "GoogleGrounded":
        return "google"
    return re.sub(r"\s+", "", provider).lower()


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


@ttl_cache(ttl=900)  # 15-min cache
def get_active_models() -> Sequence[str]:
    query = select(LanguageModel.internal_name).where(
        LanguageModel.deleted_at.is_(None),  # type: ignore
        LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
    )

    with Session(get_engine()) as session:
        active_models = session.exec(query).all()
        logging.info(f"Refreshed active models: {active_models}")
        return active_models


@ttl_cache(ttl=900)  # 15-min cache
def get_image_attachment_models() -> Sequence[str]:
    query = select(LanguageModel.internal_name).where(
        LanguageModel.supported_attachment_mime_types.contains([IMAGE_ATTACHMENT_MIME_TYPE]),  # type: ignore
        LanguageModel.deleted_at.is_(None),  # type: ignore
        LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
    )

    with Session(get_engine()) as session:
        image_attachment_models = session.exec(query).all()
        logging.info(f"Refreshed image attachment models: {image_attachment_models}")
        return image_attachment_models


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


def get_preferences(user_id: str | None, chat_id: str) -> tuple[RoutingPreference, list[str]]:
    """Returns the preferences and user-selected models for the given chat ID."""
    # Note that right now we don't use user_id in SQL query or forcing it to match the chat_id,
    # but we can do it after the frontend is updated for more security.
    sql_query = text(
        """
        SELECT cm.assistant_model_name, t.turn_id, t.creator_user_id,
               e.eval_type, me.score, cm.assistant_selection_source
        FROM turns t
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
        model_name, turn_id, creator_user_id, eval_type, score, selection_src = row
        df_rows.append(
            dict(
                model_name=model_name,
                turn_id=turn_id,
                creator_user_id=creator_user_id,
                eval_type=eval_type,
                score=score,
                selection_src=selection_src,
            )
        )

    if not df_rows:
        return RoutingPreference(turns=[], user_id=user_id), []

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

    user_id = df["creator_user_id"].tolist()[0]

    return RoutingPreference(turns=preferred_models_list, user_id=user_id), user_selected_models


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


OPENAI_FT_ID_PATTERN = re.compile(r"^ft:(?P<model>.+?):(?P<organization>.+?)::(?P<id>.+?)$")


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
