import asyncio
import logging
import uuid

from sqlalchemy import delete
from sqlmodel import select

from ypl.backend.db import get_async_session
from ypl.backend.llm.context import get_curated_chat_context
from ypl.backend.llm.judge import ConversationStartersLabeler, SuggestedFollowupsLabeler, SuggestedPromptboxLabeler
from ypl.backend.llm.provider.provider_clients import get_internal_provider_client
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import Chat, SuggestedPromptType, SuggestedTurnPrompt, SuggestedUserPrompt

MAX_TOKENS = 1000


async def maybe_add_suggested_followups(chat_id: uuid.UUID, turn_id: uuid.UUID) -> None:
    try:
        chat_context = await get_curated_chat_context(
            chat_id,
            use_all_models_in_chat_history=True,
            model="",
            current_turn_id=turn_id,
            include_current_turn=True,
            max_turns=5,
            max_message_length=1000,
            context_for_logging="add_suggested_followups",
        )

        llm = await get_internal_provider_client("gpt-4o-mini", max_tokens=MAX_TOKENS)
        followup_labeler = SuggestedFollowupsLabeler(llm, timeout_secs=3)
        promptbox_labeler = SuggestedPromptboxLabeler(llm, timeout_secs=3)
        suggested_followups, suggested_placeholder = await asyncio.gather(
            followup_labeler.alabel(chat_context.messages),
            promptbox_labeler.alabel(chat_context.messages),
        )
        suggested_turn_prompts = []
        if suggested_followups:
            suggested_turn_prompts.extend(
                [
                    SuggestedTurnPrompt(
                        turn_id=turn_id,
                        prompt=followup["suggestion"],
                        summary=followup["label"],
                        suggestion_type=SuggestedPromptType.FOLLOWUP,
                    )
                    for followup in suggested_followups
                ]
            )
            if suggested_placeholder:
                suggested_turn_prompts.append(
                    SuggestedTurnPrompt(
                        turn_id=turn_id,
                        prompt=suggested_placeholder,
                        summary=suggested_placeholder,
                        suggestion_type=SuggestedPromptType.PROMPTBOX_PLACEHOLDER,
                    )
                )

        logging.info(
            json_dumps(
                {
                    "message": "Suggested follow ups",
                    "turn_id": turn_id,
                    "suggested_followups": suggested_followups,
                    "suggested_promptbox_placeholder": suggested_placeholder,
                }
            )
        )
        if not suggested_turn_prompts:
            return

        async with get_async_session() as session:
            # Delete existing suggestions for this turn -- the new ones have more context (i.e., after "show me more").
            delete_query = delete(SuggestedTurnPrompt).where(SuggestedTurnPrompt.turn_id == turn_id)  # type: ignore
            await session.exec(delete_query)  # type: ignore
            session.add_all(suggested_turn_prompts)
            await session.commit()
    except Exception as e:
        logging.error(f"Error adding suggested followups: {e}")


async def refresh_conversation_starters(
    user_id: str,
    max_recent_chats: int = 10,
    max_turns_per_chat: int = 10,
    max_message_length: int = 1000,
    min_new_chats: int = 2,
) -> None:
    """Refresh conversation starters for a user.

    Args:
        user_id: The user ID to refresh conversation starters for.
        max_recent_chats: The maximum number of recent chats to consider.
        max_messages_per_chat: The maximum number of messages to consider per chat.
        max_message_length: The maximum message length. Longer messages are truncated.
        min_new_chats: The minimum number of new chats to consider.
    """
    try:
        async with get_async_session() as session:
            # The last time we refreshed conversation starters.
            last_refresh_time = (
                await session.exec(
                    select(SuggestedUserPrompt.created_at)
                    .where(
                        SuggestedUserPrompt.user_id == user_id,
                        SuggestedUserPrompt.deleted_at.is_(None),  # type: ignore
                    )
                    .order_by(SuggestedUserPrompt.created_at.desc())  # type: ignore
                    .limit(1)
                )
            ).first()

            # Get the most recent chats for this user.
            chats = (
                await session.exec(
                    select(Chat.chat_id, Chat.created_at)
                    .where(Chat.creator_user_id == user_id, Chat.deleted_at.is_(None))  # type: ignore
                    .order_by(Chat.created_at.desc())  # type: ignore
                    .limit(max_recent_chats)
                )
            ).all()

            # Check if there at least `min_new_chats`` new chats since the last refresh.
            if last_refresh_time:
                num_new_chats = len([id for id, created_at in chats if created_at > last_refresh_time])  # type: ignore
            else:
                num_new_chats = len(chats)

            if num_new_chats < min_new_chats:
                logging.info(
                    json_dumps(
                        {
                            "message": "Not enough new chats to refresh conversation starters",
                            "user_id": user_id,
                            "last_refresh_time": last_refresh_time,
                            "num_new_chats": num_new_chats,
                        }
                    )
                )
                return

            # Build the full chat context for all the chats to pass to the labeler.
            full_chat_context = []
            for chat_id, _ in chats:
                chat_context = await get_curated_chat_context(
                    chat_id,
                    use_all_models_in_chat_history=True,
                    model="",
                    max_turns=max_turns_per_chat,
                    max_message_length=max_message_length,
                    context_for_logging="refresh_conversation_starters",
                )
                full_chat_context.append(chat_context.messages)

            # Actually get conversation starters.
            labeler = ConversationStartersLabeler(
                await get_internal_provider_client("gpt-4o-mini", max_tokens=MAX_TOKENS),
                timeout_secs=15,
            )
            conversation_starters = await labeler.alabel(full_chat_context)

            if not conversation_starters:
                logging.info(
                    json_dumps(
                        {
                            "message": "No conversation starters generated",
                            "user_id": user_id,
                        }
                    )
                )
                return

            logging.info(
                json_dumps(
                    {
                        "message": "Conversation starters generated",
                        "user_id": user_id,
                        "conversation_starters": conversation_starters,
                    }
                )
            )

            # Refresh the DB data.
            suggested_user_prompts = [
                SuggestedUserPrompt(
                    user_id=user_id,
                    prompt=starter["suggestion"],
                    summary=starter["label"].capitalize(),
                    explanation=starter.get("explanation"),
                )
                for starter in conversation_starters
                if "suggestion" in starter and "label" in starter
            ]

            delete_query = delete(SuggestedUserPrompt).where(SuggestedUserPrompt.user_id == user_id)  # type: ignore
            await session.exec(delete_query)  # type: ignore
            session.add_all(suggested_user_prompts)
            await session.commit()
    except Exception as e:
        logging.error(f"Error refreshing conversation starters: {e}")
