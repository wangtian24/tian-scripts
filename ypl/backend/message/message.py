import logging
from dataclasses import dataclass
from datetime import datetime

from fastapi import HTTPException, status
from sqlmodel import desc, select
from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import ChatMessage, MessageType, Turn


@dataclass
class MessageResponse:
    """Response model for a single message."""

    message_id: str
    content: str
    created_at: datetime | None
    turn_id: str
    chat_id: str
    user_id: str


@dataclass
class MessagesResponse:
    """Response model for a paginated list of messages."""

    messages: list[MessageResponse]
    has_more_rows: bool


async def get_user_messages(
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> MessagesResponse:
    """Get messages for a user with pagination support."""
    try:
        async with get_async_session() as session:
            query = (
                select(ChatMessage, Turn)
                .select_from(ChatMessage)
                .join(Turn)
                .where(ChatMessage.message_type == MessageType.USER_MESSAGE)
                .order_by(desc(ChatMessage.created_at))
            )

            if user_id is not None:
                query = query.where(Turn.creator_user_id == user_id)

            query = query.offset(offset).limit(limit + 1)

            result = await session.execute(query)
            rows = result.unique().all()

            has_more_rows = len(rows) > limit
            if has_more_rows:
                rows = rows[:-1]

            log_dict = {
                "message": "Admin: Messages found",
                "user_id": user_id,
                "messages_count": len(rows),
                "limit": limit,
                "offset": offset,
                "has_more_rows": has_more_rows,
            }
            logging.info(json_dumps(log_dict))

            return MessagesResponse(
                messages=[
                    MessageResponse(
                        message_id=str(message.message_id),
                        content=message.content,
                        created_at=message.created_at,
                        turn_id=str(turn.turn_id),
                        chat_id=str(turn.chat_id),
                        user_id=turn.creator_user_id,
                    )
                    for message, turn in rows
                ],
                has_more_rows=has_more_rows,
            )

    except Exception as e:
        log_dict = {
            "message": "Admin: Unexpected error getting messages",
            "user_id": user_id,
            "limit": limit,
            "offset": offset,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from e
