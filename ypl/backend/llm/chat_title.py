import asyncio
import logging
import uuid

from sqlalchemy import update
from sqlalchemy.orm import load_only

from ypl.backend.db import get_async_session
from ypl.backend.llm.chat import get_curated_chat_context, get_gpt_4o_mini_llm
from ypl.backend.llm.judge import ChatTitleLabeler
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import Chat


async def maybe_set_chat_title(chat_id: uuid.UUID, turn_id: uuid.UUID, sleep_secs: float = 0.0) -> None:
    if sleep_secs > 0.0:
        await asyncio.sleep(sleep_secs)

    async with get_async_session() as session:
        result = await session.get(Chat, chat_id, options=[load_only(Chat.title_set_by_user)])  # type: ignore
        if not result or result.title_set_by_user:
            return

    try:
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

        labeler = ChatTitleLabeler(get_gpt_4o_mini_llm(), timeout_secs=4)
        title = await labeler.alabel(chat_context)

        logging.info(
            json_dumps(
                {
                    "message": "Updated chat title",
                    "turn_id": turn_id,
                    "title": title,
                    "chat_context_length": len(chat_context),
                }
            )
        )
        if not title:
            return

        async with get_async_session() as session:
            await session.exec(update(Chat).where(Chat.chat_id == chat_id).values(title=title))  # type: ignore
            await session.commit()
    except Exception as e:
        logging.error(f"Error updating chat title for chat {chat_id}: {e}")
