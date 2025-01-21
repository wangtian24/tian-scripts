import asyncio
import logging
import time
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential

from ypl.backend.config import settings
from ypl.backend.db import get_async_engine
from ypl.backend.llm.constants import ChatProvider
from ypl.backend.llm.judge import DEFAULT_PROMPT_DIFFICULTY, YuppPromptDifficultyWithCommentLabeler
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.moderation import DEFAULT_MODERATION_RESULT, amoderate
from ypl.backend.llm.vendor_langchain_adapter import GeminiLangChainAdapter
from ypl.backend.rw_cache import TurnQualityCache
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import MessageType, TurnQuality

LLM: GeminiLangChainAdapter | None = None


def get_llm() -> GeminiLangChainAdapter:
    global LLM
    if LLM is None:
        LLM = GeminiLangChainAdapter(
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
