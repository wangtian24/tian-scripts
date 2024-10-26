import logging
import re
from uuid import UUID

from fastapi import APIRouter, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ypl.backend.config import settings
from ypl.backend.db import get_async_engine
from ypl.backend.llm.chat import ModelInfo, get_chat_model
from ypl.backend.llm.constants import ChatProvider
from ypl.backend.llm.judge import YuppPromptDifficultyLabeler
from ypl.backend.rw_cache import TurnQualityCache
from ypl.db.chats import MessageType, TurnQuality

router = APIRouter()
llm = get_chat_model(
    ModelInfo(
        provider=ChatProvider.OPENAI,
        model="gpt-4o-mini",
        api_key=settings.OPENAI_API_KEY,
    ),
    temperature=0.0,
)


@router.post("/chats/{chat_id}/turns/{turn_id}:label_quality", response_model=TurnQuality)
async def label_quality(chat_id: UUID, turn_id: UUID) -> TurnQuality:
    cache = TurnQualityCache.get_instance()
    labeler = YuppPromptDifficultyLabeler(llm)

    tq = await cache.aread(turn_id, deep=True)

    if not tq:
        async with AsyncSession(get_async_engine()) as session:
            tq = TurnQuality(
                turn_id=turn_id,
                chat_id=chat_id,
            )

            session.add(tq)
            await session.commit()
            await session.refresh(tq)
            session.expunge(tq)

    turn = tq.turn

    if turn.chat_id != chat_id:
        raise HTTPException(status_code=404, detail="Turn quality not found")

    prompt = next((m.content for m in turn.chat_messages if m.message_type == MessageType.USER_MESSAGE), None)
    responses = [m.content for m in turn.chat_messages if m.message_type == MessageType.ASSISTANT_MESSAGE]

    if prompt is None:
        raise HTTPException(status_code=400, detail="Not enough prompts or responses to label quality")

    responses += ["", ""]  # ensure at least two responses

    try:
        response_out = await labeler.alabel((prompt,) + tuple(responses[:2]))  # type: ignore[arg-type]
        prompt_difficulty = int(re.search(r"\"overall\":\s*(\d+)", response_out).group(1))  # type: ignore[union-attr]
    except Exception as e:
        logging.exception(f"Error labeling prompt difficulty: {e} with turn ID {turn_id}")
        raise HTTPException(status_code=500, detail=f"Error labeling prompt difficulty: {e}") from e

    tq.prompt_difficulty = prompt_difficulty

    try:
        cache.write(key=turn_id, value=tq)
    except Exception as e:
        logging.exception(f"Error writing turn quality to cache: {e} with turn ID {turn_id}")
        raise HTTPException(status_code=500, detail=f"Error writing turn quality to cache: {e}") from e

    return tq


@router.get("/chats/{chat_id}/turns/{turn_id}/quality", response_model=TurnQuality)
async def get_quality(chat_id: UUID, turn_id: UUID) -> TurnQuality:
    cache = TurnQualityCache.get_instance()
    tq = await cache.aread(turn_id)

    if tq is None:
        raise HTTPException(status_code=404, detail="Turn quality not found")

    return tq
