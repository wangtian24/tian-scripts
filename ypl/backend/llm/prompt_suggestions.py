import logging
import uuid

from sqlalchemy import delete

from ypl.backend.db import get_async_session
from ypl.backend.llm.chat import get_curated_chat_context, get_gemini_15_flash_llm
from ypl.backend.llm.judge import SuggestedFollowupsLabeler
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import SuggestedTurnPrompt


async def maybe_add_suggested_followups(chat_id: uuid.UUID, turn_id: uuid.UUID) -> None:
    try:
        chat_context = await get_curated_chat_context(
            chat_id,
            use_all_models_in_chat_history=True,
            model="",
            current_turn_id=turn_id,
            include_current_turn=True,
            max_messages=10,
            max_message_length=1000,
        )

        labeler = SuggestedFollowupsLabeler(get_gemini_15_flash_llm())
        suggested_followups = labeler.label(chat_context)

        logging.debug(
            json_dumps(
                {
                    "message": "Suggested follow ups",
                    "turn_id": turn_id,
                    "suggested_followups": suggested_followups,
                }
            )
        )
        if not suggested_followups:
            return

        suggested_turn_prompts = [
            SuggestedTurnPrompt(
                turn_id=turn_id,
                prompt=followup["suggestion"],
                summary=followup["label"],
            )
            for followup in suggested_followups
        ]

        async with get_async_session() as session:
            # Delete existing suggestions for this turn -- the new ones have more context (i.e., after "show me more").
            delete_query = delete(SuggestedTurnPrompt).where(SuggestedTurnPrompt.turn_id == turn_id)  # type: ignore
            await session.exec(delete_query)  # type: ignore
            session.add_all(suggested_turn_prompts)
            await session.commit()
    except Exception as e:
        logging.error(f"Error adding suggested followups: {e}")
