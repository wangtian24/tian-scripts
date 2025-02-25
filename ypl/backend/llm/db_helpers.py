import logging
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from functools import lru_cache
from typing import Any
from uuid import UUID

from cachetools.func import ttl_cache
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from sqlalchemy import and_, desc, or_, text
from sqlmodel import Session, select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.db import get_async_engine, get_async_session, get_engine
from ypl.backend.llm.constants import (
    ACTIVE_MODELS_BY_PROVIDER,
    PROVIDER_MODEL_PATTERNS,
    ChatProvider,
)
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.routing.route_data_type import PreferredModel, RoutingPreference
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import (
    Chat,
    ChatMessage,
    Eval,
    EvalType,
    MessageEval,
    MessageType,
    Turn,
)
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, Provider
from ypl.db.model_promotions import ModelPromotionStatus
from ypl.db.yapps import Yapp
from ypl.utils import async_timed_cache, ifnull

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
    if len(models) == 0:
        return {}

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
    if len(models) == 0:
        return {}

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
def get_model_creation_dates(model_names: tuple[str, ...]) -> dict[str, datetime]:
    statement = (
        select(LanguageModel.internal_name, LanguageModel.created_at)
        .where(LanguageModel.internal_name.in_(model_names), LanguageModel.status == ModelPromotionStatus.ACTIVE)  # type: ignore
        .order_by(LanguageModel.created_at.asc())  # type: ignore
    )

    with Session(get_engine()) as session:
        rows = session.exec(statement).all()
    if not rows:
        return {}

    return {row[0]: row[1] for row in rows}  # type: ignore


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


def get_preferences(
    user_id: str | None, chat_id: str, turn_id: str, required_models_from_req: list[str] | None = None
) -> RoutingPreference:
    """Returns the preferences and user-selected models for the given chat ID."""

    # Use chat id to get all turns happened previously in this chat, oldest first.
    query = (
        select(  # type: ignore
            ChatMessage.assistant_model_name,
            ChatMessage.turn_sequence_number,
            Turn.turn_id,
            Turn.creator_user_id,
            Eval.eval_type,
            MessageEval.score,
            ChatMessage.assistant_selection_source,
            ChatMessage.completion_status,
        )
        .join(ChatMessage, ChatMessage.turn_id == Turn.turn_id)
        .outerjoin(MessageEval, MessageEval.message_id == ChatMessage.message_id)
        .outerjoin(Eval, Eval.eval_id == MessageEval.eval_id)
        .where(Turn.chat_id == chat_id, ChatMessage.message_type == MessageType.ASSISTANT_MESSAGE)
        .order_by(Turn.sequence_id.asc(), ChatMessage.turn_sequence_number.asc())  # type: ignore
        .limit(100)
    )
    with Session(get_engine()) as session:
        rows = session.exec(query).all()

    if not rows or len(rows) == 0:
        return RoutingPreference(turns=[], user_id=user_id, same_turn_shown_models=[])

    # Group rows by turn_id and collect info for each turn
    turns_by_id: dict[str, Any] = defaultdict(list)
    for row in rows:
        turns_by_id[str(row.turn_id)].append(row)
    turns_row_list = [(turn_id, messages) for turn_id, messages in turns_by_id.items()]

    turns_list = []
    same_turn_shown_models = []
    user_id = None

    # go through turns from oldest to newest and extract their preferred models if any.
    for cur_turn_id, messages in turns_row_list:
        shown_models = [
            msg.assistant_model_name for msg in messages if msg.completion_status and msg.completion_status.is_shown()
        ]
        failed_models = [
            msg.assistant_model_name for msg in messages if msg.completion_status and msg.completion_status.is_failure()
        ]
        preferred_models = [
            msg.assistant_model_name
            for msg in messages
            if msg.eval_type == EvalType.SELECTION and (msg.score is not None and msg.score > 99.9)
        ]
        preferred_model = preferred_models[0] if preferred_models else None  # there should be at most one
        downvoted_models = [
            msg.assistant_model_name
            for msg in messages
            if msg.eval_type == EvalType.DOWNVOTE and (msg.score is not None and msg.score < 0.1)
        ]
        has_eval = any(msg.eval_type in [EvalType.SELECTION, EvalType.DOWNVOTE, EvalType.ALL_BAD] for msg in messages)

        # store information for this turn
        turns_list.append(
            PreferredModel(
                shown_models=shown_models,
                failed_models=failed_models,
                preferred=preferred_model,
                downvoted=downvoted_models,
                has_evaluation=has_eval,
            )
        )
        # check if we are in "Show Me More" mode, where the turn_id will be the same.
        if turn_id == cur_turn_id:
            same_turn_shown_models = list(shown_models)

        if user_id is None:
            user_id = messages[0].creator_user_id

    return RoutingPreference(
        turns=turns_list,
        user_id=user_id,
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
    # NOTE: Makes blocking DB selects when there is cache miss.
    return {model: deduce_single_model_speed_score(model) for model in models}


@ttl_cache(ttl=300)  # 5 minute cache. The metrics are updated hourly in a cron job.
def deduce_single_model_speed_score(model_name: str) -> float:
    """
    Deduces the speed of a single model, which is more cache friendly.
    Returns an abstract speed score (0.0 - 1.0), which is computed based on model's
    time-to-first-token and tokens-per-second stats from the past X days.
    """

    # We estimate the 95% confidence interval for time-to-first-token p90 and tokens-per-sec p90, given the number of
    # model requests in the past X days. then we use the upper bound (a conservative value) of TTFT_p90 and TPS_p90
    # to compute the speed score.

    with Session(get_engine()) as session:
        # NOTE: Blocking DB call.
        model_info = session.exec(select(LanguageModel).where(LanguageModel.internal_name == model_name)).first()

    DEFAULT_SPEED_SCORE = 0.1
    if not model_info:
        # Unexpected. Model should be present.
        logging.info(
            json_dumps({"message": f"Model speed: {model_name} is not found, using default {DEFAULT_SPEED_SCORE}"})
        )
        return DEFAULT_SPEED_SCORE

    num_reqs = ifnull(model_info.num_requests_in_metric_window, 0)
    ttft_p90 = ifnull(model_info.first_token_p90_latency_ms, 0.0)
    ttft_p50 = ifnull(model_info.first_token_p50_latency_ms, 0.0)

    tps_p90 = ifnull(model_info.output_p90_tps, 0.0)
    tps_p50 = ifnull(model_info.output_p50_tps, 0.0)
    avg_token_count = ifnull(model_info.avg_token_count, 0.0)

    if any([num_reqs <= 0, ttft_p90 <= 0, ttft_p50 <= 0, tps_p90 <= 0, tps_p50 <= 0, avg_token_count <= 0]):
        # If any of these is zero, return default. In practice either all of them are set or none of them is set.
        # This should be rare, mainly for new models.
        logging.info(
            json_dumps({"message": f"Model speed for {model_name}: metrics are not set, using {DEFAULT_SPEED_SCORE}"})
        )
        return DEFAULT_SPEED_SCORE

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
                "message": f"Model speed: calculating for {model_name}, score = {speed_score:.3f}",
                "model": model_name,
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


async def get_chat_required_models(chat_id: UUID, turn_id: UUID) -> Sequence[str]:
    """
    Retrieves the required models for a chat from the database.
    We always only read from the first turn (sequence_id == 0)
    """
    # Read it from both the first turn and current turn for future compatibility. Right now the required_models
    # are only supported for the first turn (NEW_CHAT), but we will soon support required models mid-chat (NEW_TURNS).
    query = (
        select(Turn.required_models)
        .where(
            and_(
                Turn.required_models.is_not(None),  # type: ignore
                Turn.chat_id == chat_id,  # type: ignore
                or_(Turn.sequence_id == 0, Turn.turn_id == turn_id),  # type: ignore
            )
        )
        .order_by(desc(Turn.sequence_id))  # type: ignore
    )
    async with get_async_session() as session:
        # note that we only return the last such  (the last in time as it's sorted by sequence_id in descending order)
        return (await session.exec(query)).first() or ()


def notnull(value: str | None) -> str:
    assert value is not None, "value is required"
    return value


@ttl_cache(ttl=300)  # 5 min cache
def get_yapp_descriptions() -> dict[str, str]:
    query = (
        select(LanguageModel.internal_name, Yapp.description)
        .join(LanguageModel)
        .where(
            LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
            LanguageModel.language_model_id == Yapp.language_model_id,
        )
    )
    with Session(get_engine()) as session:
        return {row[0]: row[1] for row in session.exec(query).all()}
