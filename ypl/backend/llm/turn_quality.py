import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from typing import Any
from uuid import UUID

import numpy as np
from cachetools.func import ttl_cache
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel
from sqlalchemy import Select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from sqlmodel import Session, select
from tenacity import retry, stop_after_attempt, wait_exponential

from ypl.backend.config import settings
from ypl.backend.db import get_async_engine, get_async_session, get_engine
from ypl.backend.llm.constants import ChatProvider
from ypl.backend.llm.judge import DEFAULT_PROMPT_DIFFICULTY, YuppPromptDifficultyWithCommentLabeler
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.moderation import DEFAULT_MODERATION_RESULT, amoderate
from ypl.backend.llm.vendor_langchain_adapter import OpenAILangChainAdapter
from ypl.backend.rw_cache import TurnQualityCache
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import ChatMessage, CompletionStatus, Eval, EvalType, MessageEval, MessageType, Turn, TurnQuality
from ypl.db.language_models import LanguageModel

LLM: BaseChatModel | None = None

# Eval quality score assigned to low-quality evals.
LOW_EVAL_QUALITY_SCORE = 0.1
# Eval quality score assigned to high-quality evals.
HIGH_EVAL_QUALITY_SCORE = 1.0
# Content length buckets for response time estimation.
# For each bucket, we collect the 10th and 90th percentiles of response times seen across users.
CONTENT_LENGTH_BUCKETS = [100, 500, 1000, 5000, 10000, 50000, float("inf")]
# The number of recent evals to use for response time estimation.
NUM_RECENT_EVALS = 5000
# Any eval submitted within this time (counting from turn creation) is considered low-quality,
# regardless of the length of the responses it references.
MIN_RESPONSE_TIME_SECS = 2.5

# The score assigned to a "selected" model in a battle.
SELECTED_MODEL_SCORE = 100


STATIC_TURN_NOTES = {
    "positive_notes": [
        "On target! 🎯",
        "Fast 🔥",
        "Style ✍️",
        "Interesting 🤔",
        "Nice tone 😊",
    ],
    "negative_notes": [
        "Slow 🐌",
        "Too long 📝",
        "Not detailed ❓",
        "Off course 🙄",
        "Outdated ☎️",
    ],
}


def get_llm() -> BaseChatModel:
    global LLM
    if LLM is None:
        LLM = OpenAILangChainAdapter(
            model_info=ModelInfo(
                provider=ChatProvider.OPENAI,
                model="gpt-4o-mini",
                api_key=settings.OPENAI_API_KEY,
            ),
            model_config_=dict(
                temperature=0.0,
                max_tokens=128,
            ),
        )
    return LLM


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30), reraise=True)
async def label_turn_quality_with_retry(turn_id: UUID, chat_id: UUID, prompt: str | None = None) -> TurnQuality:
    cache = TurnQualityCache.get_instance()
    tq = await cache.aread(turn_id, deep=True)

    if not tq:
        async with AsyncSession(get_async_engine()) as session:
            tq = TurnQuality(turn_id=turn_id, chat_id=chat_id)
            session.add(tq)
            await session.commit()
            await session.refresh(tq)
            session.expunge(tq)
    else:
        # Only return if we have both moderation and difficulty results.
        if tq.prompt_difficulty is not None and tq.prompt_is_safe is not None:
            return tq

    turn = tq.turn

    if turn.chat_id != chat_id:
        raise ValueError(f"Chat ID {chat_id} mismatch with turn chat ID {turn.chat_id}")

    start_time = time.time()
    prompt = prompt or next((m.content for m in turn.chat_messages if m.message_type == MessageType.USER_MESSAGE), None)
    responses = [m.content for m in turn.chat_messages if m.message_type == MessageType.ASSISTANT_MESSAGE]

    if prompt is None:
        raise ValueError("Empty prompt, cannot label quality")

    responses += ["", ""]  # ensure at least two responses

    tasks: list[tuple[str, Any]] = []
    if tq.prompt_difficulty is None:
        labeler = YuppPromptDifficultyWithCommentLabeler(get_llm())
        tasks.append(("difficulty", labeler.alabel_full((prompt,) + tuple(responses[:2]))))

    if tq.prompt_is_safe is None:
        tasks.append(("moderate", amoderate(prompt)))

    if tasks:
        results = await asyncio.gather(*(task[1] for task in tasks), return_exceptions=True)

        for (task_type, _), result in zip(tasks, results, strict=True):
            if isinstance(result, Exception):
                log_dict = {
                    "message": f"Error in {task_type} task; using default value",
                    "turn_id": str(turn_id),
                    "error": str(result),
                }
                logging.warning(json_dumps(log_dict))
                if task_type == "difficulty":
                    prompt_difficulty = DEFAULT_PROMPT_DIFFICULTY
                elif task_type == "moderate":
                    moderation_result = DEFAULT_MODERATION_RESULT
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
            else:
                if task_type == "difficulty":
                    prompt_difficulty, prompt_difficulty_details = result  # type: ignore
                    tq.prompt_difficulty = prompt_difficulty
                    tq.prompt_difficulty_details = prompt_difficulty_details
                elif task_type == "moderate":
                    moderation_result = result  # type: ignore
                    tq.prompt_is_safe = moderation_result.safe
                    tq.prompt_moderation_model_name = moderation_result.model_name
                    if not moderation_result.safe:
                        tq.prompt_unsafe_reasons = moderation_result.reasons
                else:
                    raise ValueError(f"Unknown task type: {task_type}")

    try:
        cache.write(key=turn_id, value=tq)
    except Exception as e:
        log_dict = {
            "message": "Error writing turn quality to cache",
            "turn_id": str(turn_id),
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise ValueError(f"Error writing turn quality to cache: {e}") from e

    info = {
        "message": "Turn quality labeled",
        "turn_id": str(turn_id),
        "prompt_difficulty": tq.prompt_difficulty,
        "prompt_is_safe": tq.prompt_is_safe,
        "time_msec": (time.time() - start_time) * 1000,
    }
    logging.info(json_dumps(info))

    return tq


async def label_turn_quality(turn_id: UUID, chat_id: UUID, prompt: str | None = None) -> TurnQuality:
    try:
        return await label_turn_quality_with_retry(turn_id, chat_id, prompt)
    except Exception as e:
        logging.error(f"Failed to label turn quality after retries: {str(e)}")
        raise e


def _get_content_length_bucket(content_len: int) -> int | float:
    """Get the content length bucket for a given content length."""
    for bucket in CONTENT_LENGTH_BUCKETS:
        if content_len <= bucket:
            return bucket
    return float("inf")


@ttl_cache(ttl=36000)  # 36000 seconds = 10 hours
def get_eval_response_times_by_content_length() -> dict[int | float, tuple[float, float]]:
    """Returns the upper and lower bounds of response times for each content length bucket."""
    with Session(get_engine()) as session:
        result = session.exec(
            select(Eval)
            .join(Turn, Turn.turn_id == Eval.turn_id)  # type: ignore
            .join(ChatMessage, ChatMessage.turn_id == Turn.turn_id)  # type: ignore
            .options(joinedload(Eval.turn).joinedload(Turn.chat_messages))  # type: ignore
            .where(
                ChatMessage.message_type == MessageType.ASSISTANT_MESSAGE,
                ChatMessage.deleted_at.is_(None),  # type: ignore
                Turn.deleted_at.is_(None),  # type: ignore
                Eval.deleted_at.is_(None),  # type: ignore
                Eval.eval_type == EvalType.SELECTION,
            )
            .order_by(Eval.created_at.desc())  # type: ignore
            .limit(NUM_RECENT_EVALS)
        )
        evals = result.unique().all()

        # Group evals by content length buckets
        buckets: dict[int | float, list[float]] = {k: [] for k in CONTENT_LENGTH_BUCKETS}

        for eval in evals:
            content_length = sum(
                len(msg.content) for msg in eval.turn.chat_messages if msg.message_type == MessageType.ASSISTANT_MESSAGE
            )
            time_diff = (eval.created_at - eval.turn.created_at).total_seconds()  # type: ignore

            # Add to appropriate bucket
            bucket = _get_content_length_bucket(content_length)
            buckets[bucket].append(time_diff)

    # Calculate percentiles for each bucket
    percentiles: dict[int | float, tuple[float, float]] = {}
    for bucket_size, times in buckets.items():
        if times:
            percentiles[bucket_size] = (
                float(np.percentile(times, 10)),  # 10th percentile
                float(np.percentile(times, 90)),  # 90th percentile
            )

    logging.info(
        json_dumps(
            {
                "message": "Refreshed eval response times by content length",
                "percentiles": percentiles,
            }
        )
    )

    return percentiles


def get_eval_quality(eval: Eval) -> tuple[float, dict[str, str]]:
    """Estimate a quality score for an evaluation.

    The quality score is computed based on the response time and the length of the responses; a short response time
    implies low quality, with some adjustment for the length of the responses.

    Args:
        eval: The evaluation to score.

    Returns:
        A tuple of the the quality score and a dictionary of comments/notes about its components.
    """
    quality_score, explanation = _estimate_eval_quality_score(
        response_time_secs=(eval.created_at - eval.turn.created_at).total_seconds(),  # type: ignore
        response_lengths=tuple(
            len(m.content) for m in eval.turn.chat_messages if m.message_type == MessageType.ASSISTANT_MESSAGE
        ),
    )
    logging.info(
        json_dumps(
            {
                "message": "Eval quality",
                "turn_id": str(eval.turn.turn_id),
                "quality_score": str(quality_score),
                "explanation": explanation,
            }
        )
    )

    return quality_score, explanation


def _estimate_eval_quality_score(
    response_time_secs: float, response_lengths: tuple[int, ...]
) -> tuple[float, dict[str, str]]:
    sum_response_lengths = sum(response_lengths)
    # Estimate a minimal and maximal time the user should produce an eval in, based on the length of the responses.
    content_length_bucket = _get_content_length_bucket(sum_response_lengths)
    min_response_time_secs, max_response_time_secs = get_eval_response_times_by_content_length()[content_length_bucket]

    if (
        response_time_secs < min_response_time_secs  # too fast for this bucket of content length
        or response_time_secs < MIN_RESPONSE_TIME_SECS  # too fast for any content length
        or response_time_secs > max_response_time_secs  # too slow for this bucket of content length
    ):
        quality_score = LOW_EVAL_QUALITY_SCORE
    else:
        quality_score = HIGH_EVAL_QUALITY_SCORE

    explanation = {
        "sum_response_lengths": str(sum_response_lengths),
        "min_response_time_secs": str(min_response_time_secs),
        "max_response_time_secs": str(max_response_time_secs),
        "actual_response_time_secs": str(response_time_secs),
        "quality_score": str(quality_score),
    }

    return quality_score, explanation


async def update_user_eval_quality_scores(user_id: str) -> None:
    """Update the quality scores for all evals for a user that are missing quality scores."""
    async with get_async_session() as session:
        # Evals for this user that don't have quality scores set.
        query = (
            select(Eval)
            .options(joinedload(Eval.turn).joinedload(Turn.chat_messages))  # type: ignore
            .where(
                Eval.user_id == user_id,
                Eval.quality_score.is_(None),  # type: ignore
                Eval.deleted_at.is_(None),  # type: ignore
                Eval.eval_type == EvalType.SELECTION,
            )
        )
        result = await session.exec(query)
        evals = result.unique().all()

        logging.info(
            json_dumps(
                {
                    "message": "Updating eval quality scores for user",
                    "user_id": user_id,
                    "num_evals": len(evals),
                }
            )
        )

        for eval in evals:
            quality_score, quality_score_notes = get_eval_quality(eval)
            eval.quality_score = quality_score
            eval.quality_score_notes = quality_score_notes

        await session.commit()


class TurnEvalCount(BaseModel):
    """Counts the number of evals and wins for a given language model."""

    language_model_id: uuid.UUID
    language_model_name: str
    count: int
    unique_user_count: int
    wins: int
    unique_user_wins: int


class TurnAnnotations(BaseModel):
    """Annotations for a given turn."""

    # The LLM_judge comment on the turn.
    comment: str | None = None
    # Positive eval notes.
    positive_notes: list[tuple[str, str]] | None = None
    # Negative eval notes.
    negative_notes: list[tuple[str, str]] | None = None
    # The number of past evals and wins for each language model in the turn, by the same user as the turn's creator.
    # Eg. reports "X was preferred over Y 15 times in the past by you"
    user_eval_counts: list[TurnEvalCount] | None = None
    # The number of evals and wins for all language models in the turn by any user.
    # Eg. reports "X was preferred over Y 60 times in the past by all users"
    overall_eval_counts: list[TurnEvalCount] | None = None
    # The number of wins against any model by the turn's creator.
    # Eg. reports "X was preferred 35 times in the past by you"
    user_model_eval_counts: list[TurnEvalCount] | None = None
    # The number of wins against any model by all users.
    # Eg. reports "X was preferred 60 times in the past by all users"
    overall_model_eval_counts: list[TurnEvalCount] | None = None


def _get_evals_with_same_models_in_turn(current_turn_id: UUID, limit_to_turn_creator: bool = True) -> Select:
    # Temp table for the language models used in the turn.
    ordered_models = (
        select(ChatMessage.turn_id, ChatMessage.assistant_language_model_id)
        .where(
            ChatMessage.message_type == MessageType.ASSISTANT_MESSAGE,
            ChatMessage.deleted_at.is_(None),  # type: ignore
            (ChatMessage.completion_status == CompletionStatus.SUCCESS) | (ChatMessage.completion_status.is_(None)),  # type: ignore
            ChatMessage.assistant_language_model_id.is_not(None),  # type: ignore
        )
        .distinct()
        .order_by(ChatMessage.turn_id, ChatMessage.assistant_language_model_id)  # type: ignore
    ).cte("ordered_models")

    models_by_turn = (
        select(
            ordered_models.c.turn_id, func.array_agg(ordered_models.c.assistant_language_model_id).label("models")
        ).group_by(ordered_models.c.turn_id)
    ).cte("models_by_turn")

    current_turn_models = (
        select(models_by_turn.c.models).where(models_by_turn.c.turn_id == current_turn_id).scalar_subquery()
    )

    # The actual query.
    # Get the evals of any model in the turn.
    query = _get_evals_with_any_model_in_turn(current_turn_id, limit_to_turn_creator)
    # Filter the evals to only include those that exactly match the models in the turn.
    query = query.join(models_by_turn, models_by_turn.c.turn_id == Eval.turn_id).where(
        models_by_turn.c.models == current_turn_models
    )

    return query


def _get_evals_with_any_model_in_turn(current_turn_id: UUID, limit_to_turn_creator: bool = True) -> Select:
    """Returns a query to fetche all evals (or just those by the turn's creator) of any model in the turn."""
    query = (
        select(MessageEval.eval_id, MessageEval.score, ChatMessage.assistant_language_model_id, Eval.user_id)
        .join(Eval, Eval.eval_id == MessageEval.eval_id)  # type: ignore
        .join(ChatMessage, ChatMessage.message_id == MessageEval.message_id)  # type: ignore
        .where(
            Eval.eval_type == EvalType.SELECTION,
            (ChatMessage.completion_status == CompletionStatus.SUCCESS) | (ChatMessage.completion_status.is_(None)),  # type: ignore
            ChatMessage.assistant_language_model_id.in_(  # type: ignore
                select(ChatMessage.assistant_language_model_id)
                .where(ChatMessage.turn_id == current_turn_id)
                .distinct()
                .scalar_subquery()
            ),
            MessageEval.score.is_not(None),  # type: ignore
        )
    )
    if limit_to_turn_creator:
        query = query.where(
            Eval.user_id == (select(Turn.creator_user_id).where(Turn.turn_id == current_turn_id).scalar_subquery())
        )
    return query


async def _aggregate_evals(session: AsyncSession, query: Select) -> list[TurnEvalCount]:
    """Aggregate the evals, into a list of TurnEvalCount objects, one per model."""
    result = await session.execute(query)
    rows = result.fetchall()

    evals_by_model = defaultdict(list)
    unique_users_by_model = defaultdict(set)
    wins_by_model = defaultdict(set)
    unique_user_wins_by_model = defaultdict(set)
    for eval_id, score, assistant_language_model_id, user_id in rows:
        evals_by_model[assistant_language_model_id].append(eval_id)
        unique_users_by_model[assistant_language_model_id].add(user_id)
        if score == SELECTED_MODEL_SCORE:
            wins_by_model[assistant_language_model_id].add(eval_id)
            unique_user_wins_by_model[assistant_language_model_id].add(user_id)

    # Get all model names in a single query
    result = await session.execute(
        select(LanguageModel.language_model_id, LanguageModel.name).where(
            LanguageModel.language_model_id.in_(evals_by_model.keys())  # type: ignore
        )
    )
    model_names: dict[uuid.UUID, str] = dict((row[0], row[1]) for row in result.fetchall())

    return [
        TurnEvalCount(
            language_model_id=model_id,
            language_model_name=model_names[model_id],
            count=len(evals_by_model[model_id]),
            unique_user_count=len(unique_users_by_model[model_id]),
            wins=len(wins_by_model[model_id]),
            unique_user_wins=len(unique_user_wins_by_model[model_id]),
        )
        for model_id in evals_by_model
    ]


def _get_turn_notes(details_dict: dict[str, Any], attr: str, max_notes: int = 5) -> list[tuple[str, str]]:
    """Returns the first 2 static notes of type `attr`, followed by the dynamic notes, up to `max_notes`."""
    static_notes = STATIC_TURN_NOTES.get(attr, [])
    dynamic_notes = details_dict.get(attr, [])
    notes = static_notes[:2] + dynamic_notes + static_notes[2:]
    seen = set()
    notes = [note for note in notes if not (note in seen or seen.add(note))]  # type: ignore
    return [tuple(n.rsplit(maxsplit=1)) for n in notes[:max_notes]]


async def get_turn_annotations(turn_id: UUID) -> TurnAnnotations:
    async with get_async_session() as session:
        annotations = TurnAnnotations()
        stmt = select(TurnQuality.prompt_difficulty_details).where(TurnQuality.turn_id == turn_id)
        result = await session.execute(stmt)
        details = result.scalar_one_or_none()
        if details:
            details_dict = json.loads(details)
            annotations.comment = details_dict.get("comment")
            for attr in ("positive_notes", "negative_notes"):
                if attr in details_dict and attr in STATIC_TURN_NOTES:
                    setattr(annotations, attr, _get_turn_notes(details_dict, attr))

        # Add past battle results for the same user and for all users.
        same_user_same_models_query = _get_evals_with_same_models_in_turn(turn_id, limit_to_turn_creator=True)
        annotations.user_eval_counts = await _aggregate_evals(session, same_user_same_models_query)
        all_users_same_models_query = _get_evals_with_same_models_in_turn(turn_id, limit_to_turn_creator=False)
        annotations.overall_eval_counts = await _aggregate_evals(session, all_users_same_models_query)

        # Add total eval counts for all models in the turn.
        same_user_any_model_in_turn_query = _get_evals_with_any_model_in_turn(turn_id, limit_to_turn_creator=True)
        annotations.user_model_eval_counts = await _aggregate_evals(session, same_user_any_model_in_turn_query)
        all_users_any_model_in_turn_query = _get_evals_with_any_model_in_turn(turn_id, limit_to_turn_creator=False)
        annotations.overall_model_eval_counts = await _aggregate_evals(session, all_users_any_model_in_turn_query)

        return annotations
