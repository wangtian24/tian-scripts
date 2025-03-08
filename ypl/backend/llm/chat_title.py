import asyncio
import logging
import uuid

from sqlalchemy import func, select, update
from sqlalchemy.orm import load_only

from ypl.backend.db import get_async_session
from ypl.backend.llm.context import get_curated_chat_context
from ypl.backend.llm.judge import ChatTitleLabeler
from ypl.backend.llm.provider.provider_clients import get_internal_provider_client
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import Chat, Turn

MAX_TOKENS = 64
UPDATE_EVERY_N_TURNS = 10


async def get_chat_title_suggestion(chat_id: uuid.UUID, turn_id: uuid.UUID | None = None) -> str:
    chat_context = await get_curated_chat_context(
        chat_id,
        use_all_models_in_chat_history=True,
        model="",
        current_turn_id=turn_id,
        include_current_turn=True,
        max_turns=10,
        max_message_length=1000,
        context_for_logging="set_chat_title",
    )
    if not chat_context or not chat_context.messages:
        raise ValueError("No chat context or messages")

    labeler = ChatTitleLabeler(
        await get_internal_provider_client("gemini-2.0-flash-001", max_tokens=MAX_TOKENS), timeout_secs=4
    )
    title = await labeler.alabel(chat_context.messages)
    logging.info(
        json_dumps(
            {
                "message": "Suggesting chat title",
                "chat_id": chat_id,
                "turn_id": turn_id,
                "title": title,
                "chat_context_length": len(chat_context.messages),
            }
        )
    )
    return title


async def maybe_set_chat_title(chat_id: uuid.UUID, turn_id: uuid.UUID, sleep_secs: float = 0.0) -> None:
    if sleep_secs > 0.0:
        await asyncio.sleep(sleep_secs)

    async with get_async_session() as session:
        result = await session.get(Chat, chat_id, options=[load_only(Chat.title_set_by_user)])  # type: ignore
        if not result or result.title_set_by_user:
            return

        turn_count = await session.scalar(select(func.count()).select_from(Turn).where(Turn.chat_id == chat_id))  # type: ignore

    if turn_count and turn_count % UPDATE_EVERY_N_TURNS != 1:
        return

    logging.info(f"Updating chat title for chat {chat_id} at turn number {turn_count}")

    try:
        title = await get_chat_title_suggestion(chat_id, turn_id)

        if not title:
            return

        async with get_async_session() as session:
            await session.exec(update(Chat).where(Chat.chat_id == chat_id).values(title=title))  # type: ignore
            await session.commit()
    except Exception as e:
        logging.error(f"Error updating chat title for chat {chat_id}: {e}")
