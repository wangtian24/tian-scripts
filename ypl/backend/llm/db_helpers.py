import logging
import math
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
from ypl.backend.llm.constants import (
    ACTIVE_MODELS_BY_PROVIDER,
    PROVIDER_MODEL_PATTERNS,
    ChatProvider,
)
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.routing.route_data_type import PreferredModel, RoutingPreference
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import Chat, ChatMessage, MessageType, Turn
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, Provider
from ypl.utils import async_timed_cache

IMAGE_ATTACHMENT_MIME_TYPE = "image/*"
PDF_ATTACHMENT_MIME_TYPE = "application/pdf"


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
                break

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


@ttl_cache(ttl=3600)  # 1 hour cache
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


@ttl_cache(ttl=3600)  # 1 hour cache
def get_pdf_attachment_models() -> Sequence[str]:
    query = select(LanguageModel.internal_name).where(
        LanguageModel.supported_attachment_mime_types.contains([PDF_ATTACHMENT_MIME_TYPE]),  # type: ignore
        LanguageModel.deleted_at.is_(None),  # type: ignore
        LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
    )

    with Session(get_engine()) as session:
        pdf_attachment_models = session.exec(query).all()
        logging.info(f"Refreshed pdf attachment models: {pdf_attachment_models}")
        return pdf_attachment_models


@ttl_cache(ttl=86400)  # 1 day cache
def get_model_context_lengths() -> dict[str, int]:
    query = select(LanguageModel.internal_name, LanguageModel.context_window_tokens).where(
        LanguageModel.deleted_at.is_(None),  # type: ignore
        LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
    )

    with Session(get_engine()) as session:
        results = session.exec(query).all()
        return {name: length for name, length in results if length is not None}


def get_preferences(user_id: str | None, chat_id: str, turn_id: str) -> RoutingPreference:
    """Returns the preferences and user-selected models for the given chat ID."""

    # Use chat id to get all turns happened previously in this chat, oldest first.
    sql_query = text(
        """
        SELECT cm.assistant_model_name, t.turn_id, t.creator_user_id,
               e.eval_type, me.score, cm.assistant_selection_source
        FROM turns t
            JOIN chat_messages cm ON cm.turn_id = t.turn_id
            LEFT JOIN message_evals me ON me.message_id = cm.message_id
            LEFT JOIN evals e ON e.eval_id = me.eval_id
        WHERE t.chat_id = :chat_id AND cm.message_type = 'ASSISTANT_MESSAGE'
        ORDER BY cm.created_at ASC
        LIMIT 100
        """
    )

    with get_engine().connect() as conn:
        c = conn.execute(sql_query, dict(chat_id=chat_id))
        rows = c.fetchall()

    df_rows = []

    for row in rows:
        model_name, db_turn_id, creator_user_id, eval_type, score, selection_src = row
        df_rows.append(
            dict(
                model_name=model_name,
                turn_id=str(db_turn_id),
                creator_user_id=str(creator_user_id),
                eval_type=eval_type,
                score=score,
                selection_src=selection_src,
            )
        )

    if not df_rows:
        return RoutingPreference(turns=[], user_selected_models=[], user_id=user_id, same_turn_shown_models=[])

    df = pd.DataFrame(df_rows)
    turns_list = []
    same_turn_shown_models = []

    # go through turns from oldest to newest and extract their preferred models if any.
    for _, gdf in df.groupby("turn_id", sort=False):
        cur_turn_id = str(gdf["turn_id"].tolist()[0])
        all_models = gdf["model_name"].tolist()

        evaluated_models = gdf[gdf["eval_type"].isin(["SELECTION", "DOWNVOTE", "ALL_BAD"])]["model_name"].tolist()
        has_eval = len(evaluated_models) > 0

        preferred_models = gdf[(gdf["eval_type"] == "SELECTION") & (gdf["score"] == 100.0)]["model_name"].tolist()
        preferred_model = preferred_models[0] if preferred_models else None

        # store information for this turn
        turns_list.append(PreferredModel(models=all_models, preferred=preferred_model, has_evaluation=has_eval))

        # check if we are in "Show Me More" mode, where the turn_id will be the same.
        if turn_id == cur_turn_id:
            same_turn_shown_models = list(all_models)

    # Collect user-selected models from all turns. this way to infer the user-selected models is not ideal, as
    # not all user-selected models are necessarily shown before (if we allow more user selected models than we display).
    user_selected_models = list(dict.fromkeys(df[df["selection_src"] == "USER_SELECTED"]["model_name"].tolist()))

    # NOTE(Tian): this is a bit hacky, but it guarantees the last-turn preferred model will be shown,
    # if there's space for it, it can show up in SMM, but will it be recorded as a USER_SELECTED model in db?
    if len(turns_list) > 0 and turns_list[-1].preferred:
        user_selected_models.append(turns_list[-1].preferred)

    user_id = df["creator_user_id"].tolist()[0]

    return RoutingPreference(
        turns=turns_list,
        user_id=user_id,
        user_selected_models=user_selected_models,
        same_turn_shown_models=same_turn_shown_models,
    )


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


def deduce_model_speed_scores(models: tuple[str, ...]) -> dict[str, float]:
    return {model: deduce_single_model_speed_score(model) for model in models}


@ttl_cache(ttl=7200)  # 2-hr cache, this doesn't change that fast
def deduce_single_model_speed_score(model: str) -> float:
    """
    Deduces the speed of a single model, which is more cache friendly.
    Returns an abstract speed score (0.0 - 1.0), which is computed based on model's
    time-to-first-token and tokens-per-second stats from the past X days.
    """

    # TODO(Tian): we are taking past 1000 requests for each model to estimate the speed, but it would be
    # better to estimate a short-term speed and long-term speed separately, so we can be responsive to some
    # temporary performance issues (like a certain API was having a bad day). Also this speed score is currently
    # only used to rank last 2 models displayed, not to rank all models proposals in routing, which we should
    # eventually do.
    sql_query = text(
        """
            select
                COUNT(1) as requests,
                PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY ttft) AS ttft_p90,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ttft) AS ttft_p50,
                PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY tps) AS tps_p90,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tps) AS tps_p50,
                AVG(tokenCount) AS avg_token_count
            FROM (
                SELECT
                    message_type,
                    assistant_language_model_id,
                    created_at,
                    firstToken,
                    (firsttoken - requesttimestamp) AS ttft,
                    CASE
                        WHEN tokenCount IS NULL OR tokenCount = 0 THEN 0
                        ELSE tokenCount / (lasttoken - requesttimestamp) * 1000
                    END as tps,
                    tokenCount
                FROM (
                    SELECT
                        (streaming_metrics->>'requestTimestamp')::DECIMAL AS requestTimestamp,
                        (streaming_metrics->>'firstTokenTimestamp')::DECIMAL AS firstToken,
                        (streaming_metrics->>'lastTokenTimestamp')::DECIMAL AS lastToken,
                        (streaming_metrics->>'completionTokens')::DECIMAL AS tokenCount,
                        message_type,
                        assistant_language_model_id,
                        cm.created_at
                    FROM chat_messages cm
                        JOIN language_models lm ON cm.assistant_model_name = lm.internal_name
                    WHERE message_type IN ('ASSISTANT_MESSAGE')
                        AND streaming_metrics IS NOT NULL
                        AND lm.internal_name = :model_name
                    ORDER BY cm.created_at DESC
                    LIMIT 1000
                ) AS subquery
            ) AS metrics
            WHERE ttft > 0 and tps > 0
        """
    )

    # We estimate the 95% confidence interval for time-to-first-token p90 and tokens-per-sec p90, given the number of
    # model requests in the past X days. then we use the upper bound (a conservative value) of TTFT_p90 and TPS_p90
    # to compute the speed score.

    with get_engine().connect() as conn:
        c = conn.execute(sql_query, dict(model_name=model))
        rows = c.fetchall()

    DEFAULT_SPEED_SCORE = 0.1
    if not rows:
        logging.info(
            json_dumps(
                {"message": f"Model speed: calculating for {model}, no stats, using default {DEFAULT_SPEED_SCORE}"}
            )
        )
        return DEFAULT_SPEED_SCORE

    row = rows[0]
    if any(x is None for x in row):
        logging.info(
            json_dumps(
                {"message": f"Model speed: calculating for {model}, bad stats, using default {DEFAULT_SPEED_SCORE}"}
            )
        )
        return DEFAULT_SPEED_SCORE

    num_reqs, ttft_p90, ttft_p50 = row[0], float(row[1]), float(row[2])
    tps_p90, tps_p50, avg_token_count = float(row[3]), float(row[4]), float(row[5])

    Z95 = 1.96  # z-score for 95% confidence interval

    # estimate CI for TTFT, assume log-normal distribution as this is a latency
    std = max(0.1, (math.log(ttft_p90) - math.log(ttft_p50))) / 1.28  # standard deviation
    se = std / math.sqrt(num_reqs)  # standard error
    log_ttft_p90_ub = math.log(ttft_p90) + Z95 * se  # upperbound of 95% confidence interval (z_0.025 = 1.96)
    ttft_p90_ub = math.exp(log_ttft_p90_ub)  # upperbound of 95% confidence interval

    # estimate CI for TPS, assume normal distribution, this is a speed
    std = max(0.1, (tps_p90 - tps_p50)) / 1.28  # standard deviation
    fp = 1.0 / (math.sqrt(2 * math.pi) * std)
    se = math.sqrt(0.9 * (1 - 0.9) / (num_reqs * fp**2))  # standard error
    tps_p90_lb = tps_p90 - Z95 * se  # lowerbound of 95% confidence interval

    est_streaming_latency = avg_token_count / tps_p90_lb * 1000  # time to generate the whole sequence

    # the overall latency estimate is a weighted sume of TFTT and TPS.
    TTFT_WEIGHT = 0.6
    STREAMING_WEIGHT = 0.4
    latency_est = ttft_p90_ub * TTFT_WEIGHT + est_streaming_latency * STREAMING_WEIGHT
    speed_score = 1000.0 / (1000.0 + latency_est)

    logging.info(
        json_dumps(
            {
                "message": f"Model speed: calculating for {model}, score = {speed_score:.3f}",
                "model": model,
                "num_requests": num_reqs,
                "ttft_p90_ms": round(ttft_p90, 2),
                "ttft_p90_ub_ms": round(ttft_p90_ub, 2),
                "tps_p90": round(tps_p90, 2),
                "tps_p90_lb": round(tps_p90_lb, 2),
                "avg_token_count": round(avg_token_count, 2),
                "est_streaming_latency_ms": round(est_streaming_latency, 2),
                "speed_score": round(speed_score, 3),
            }
        )
    )

    return speed_score


def get_chat_required_models(chat_id: UUID) -> Sequence[str]:
    """
    Retrieves the required models for a chat from the database.
    We always only read from the first turn (sequence_id == 0)
    """
    # TODO(Tian): later we can read it from later turns as well, but from offline discussion the FE will
    # always pass them in.
    query = select(Turn.required_models).where(Turn.chat_id == chat_id, Turn.sequence_id == 0)
    with Session(get_engine()) as session:
        return session.exec(query).first() or ()
