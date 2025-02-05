from datetime import datetime
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import load_only, selectinload
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.db.chats import Chat, ChatMessage, MessageType, Turn

DEFAULT_MESSAGE_TYPES = (MessageType.ASSISTANT_MESSAGE, MessageType.USER_MESSAGE, MessageType.QUICK_RESPONSE_MESSAGE)
DEFAULT_MESSAGE_FIELDS = ("created_at", "content", "message_type")
EPSILON = 1e-9


async def search_chats(
    session: AsyncSession,
    query: str,
    limit: int = 20,
    offset: int = 0,
    order_by: Literal["relevance", "created_at"] = "created_at",
    message_types: tuple[MessageType, ...] | None = None,
    creator_user_id: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[Chat]:
    """
    Searches for chats, matching content in any of the messages in any turn in the chat.

    Search is performed in two steps:
      1. Filter to chats that have at least one matching message using an OR condition so that even if individual
         messages only cover a subset of the query terms, the chat is still a candidate for a full match.
      2. Aggregate the content_tsvectors for all messages in the chat and ranks using the full vector.
    """
    message_types = message_types or DEFAULT_MESSAGE_TYPES

    # We con't yet support full boolean format.
    cleaned_query = query.translate(str.maketrans("", "", ":()&|!\"'"))
    query = cleaned_query.strip()

    # Filters that will be reused in the second step.
    filters = [
        ChatMessage.message_type.in_(message_types),  # type: ignore
        Chat.deleted_at.is_(None),  # type: ignore
    ]
    if creator_user_id:
        filters.append(Chat.creator_user_id == creator_user_id)
    if start_date:
        filters.append(Chat.created_at >= start_date)  # type: ignore
    if end_date:
        filters.append(Chat.created_at <= end_date)  # type: ignore

    # Find messages that match at least one term by making an OR query with all terms.
    tokens = set(query.split())
    or_query_str = " | ".join(tokens) if tokens else ""
    ts_query_filter = func.to_tsquery(or_query_str) if or_query_str else None

    candidates = (
        select(Chat.chat_id)  # type: ignore
        .join(Turn, Turn.chat_id == Chat.chat_id)
        .join(ChatMessage, ChatMessage.turn_id == Turn.turn_id)
        .filter(*filters)
        .filter(ChatMessage.content_tsvector.op("@@")(ts_query_filter))  # type: ignore
        .distinct()
        .cte("candidates")
    )

    # Create a tsquery for all terms in the query.
    ts_query_rank = func.plainto_tsquery(query)

    # Combine all content_tsvectors for all messages in a chat into a single searchable tsvector.
    aggregated_tsvector = func.tsvector_agg(ChatMessage.content_tsvector)
    score = func.ts_rank(aggregated_tsvector, ts_query_rank)

    # Rank all candidate chats using the aggregated tsvectors.
    stmt = (
        select(Chat)
        .join(Turn, Turn.chat_id == Chat.chat_id)  # type: ignore
        .join(ChatMessage, ChatMessage.turn_id == Turn.turn_id)  # type: ignore
        .join(candidates, candidates.c.chat_id == Chat.chat_id)
        .filter(*filters)
        .group_by(Chat.chat_id)  # type: ignore
        .having(score > EPSILON)
        .limit(limit)
        .offset(offset)
        .options(selectinload(Chat.turns))  # type: ignore
    )
    if order_by == "relevance":
        stmt = stmt.order_by(score.desc())
    elif order_by == "created_at":
        stmt = stmt.order_by(Chat.created_at.desc())  # type: ignore

    results = await session.exec(stmt)  # type: ignore
    return results.scalars().all()  # type: ignore


async def search_chat_messages(
    session: AsyncSession,
    query: str,
    limit: int = 20,
    offset: int = 0,
    order_by: Literal["relevance", "created_at"] = "created_at",
    message_types: tuple[MessageType, ...] | None = None,
    creator_user_id: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    message_fields: tuple[str, ...] | None = None,  # Which field to hydrate in a ChatMessage
) -> list[ChatMessage]:
    """Searches for chat messages."""
    message_types = message_types or DEFAULT_MESSAGE_TYPES
    message_fields = message_fields or DEFAULT_MESSAGE_FIELDS

    message_attrs = []
    for field in message_fields:
        if not hasattr(ChatMessage, field):
            raise AttributeError(f"Unknown field '{field}'")
        message_attrs.append(getattr(ChatMessage, field))

    filters = [
        ChatMessage.message_type.in_(message_types),  # type: ignore
        Chat.deleted_at.is_(None),  # type: ignore
    ]
    if creator_user_id:
        filters.append(Turn.creator_user_id == creator_user_id)
    if start_date:
        filters.append(ChatMessage.created_at >= start_date)  # type: ignore
    if end_date:
        filters.append(ChatMessage.created_at <= end_date)  # type: ignore

    ts_query = func.plainto_tsquery(query)
    score = func.ts_rank(ChatMessage.content_tsvector, ts_query)
    filters.append(ChatMessage.content_tsvector.op("@@")(ts_query))  # type: ignore

    stmt = (
        select(ChatMessage).options(load_only(*message_attrs)).join(Turn).filter(*filters).limit(limit).offset(offset)
    )

    if order_by == "relevance":
        stmt = stmt.order_by(score.desc())
    elif order_by == "created_at":
        stmt = stmt.order_by(ChatMessage.created_at.desc())  # type: ignore

    results = await session.exec(stmt)  # type: ignore
    return results.scalars().all()  # type: ignore
