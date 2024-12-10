import asyncio
import logging
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from tenacity import retry, stop_after_attempt, wait_exponential

from ypl.backend.config import settings
from ypl.backend.db import get_async_engine
from ypl.backend.llm.chat import ModelInfo, get_chat_model
from ypl.backend.llm.constants import ChatProvider
from ypl.backend.llm.judge import (
    DEFAULT_PROMPT_DIFFICULTY,
    YuppPromptDifficultyLabelerSimple,
)
from ypl.backend.llm.moderation import DEFAULT_MODERATION_RESULT, amoderate
from ypl.backend.rw_cache import TurnQualityCache
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import Chat, ChatMessage, MessageType, TurnQuality

router = APIRouter()
llm = get_chat_model(
    ModelInfo(
        provider=ChatProvider.OPENAI,
        model="gpt-4o",
        api_key=settings.OPENAI_API_KEY,
    ),
    temperature=0.0,
)

MAX_PINNED_CHATS = 10


@router.post("/chats/{chat_id}/turns/{turn_id}:label_quality", response_model=TurnQuality)
async def label_quality(chat_id: UUID, turn_id: UUID) -> TurnQuality:
    cache = TurnQualityCache.get_instance()
    labeler = YuppPromptDifficultyLabelerSimple(llm)

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

    label_task = asyncio.create_task(labeler.alabel((prompt,) + tuple(responses[:2])))  # type: ignore
    moderate_task = asyncio.create_task(amoderate(prompt))

    try:
        prompt_difficulty: int = await label_task
    except Exception as e:
        log_dict = {
            "message": "Error labeling prompt difficulty; assigning default value",
            "turn_id": str(turn_id),
            "error": str(e),
        }
        logging.warning(json_dumps(log_dict))
        prompt_difficulty = DEFAULT_PROMPT_DIFFICULTY

    try:
        moderation_result = await moderate_task
    except Exception as e:
        log_dict = {
            "message": "Error getting moderation result; assigning default value",
            "turn_id": str(turn_id),
            "error": str(e),
        }
        logging.warning(json_dumps(log_dict))
        moderation_result = DEFAULT_MODERATION_RESULT

    tq.prompt_difficulty = prompt_difficulty
    tq.prompt_is_safe = moderation_result.safe
    tq.prompt_moderation_model_name = moderation_result.model_name
    if not moderation_result.safe:
        tq.prompt_unsafe_reasons = moderation_result.reasons

    try:
        cache.write(key=turn_id, value=tq)
    except Exception as e:
        log_dict = {
            "message": "Error writing turn quality to cache",
            "turn_id": str(turn_id),
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=f"Error writing turn quality to cache: {e}") from e

    return tq


@router.get("/chats/{chat_id}/turns/{turn_id}/quality", response_model=TurnQuality)
async def get_quality(chat_id: UUID, turn_id: UUID) -> TurnQuality:
    cache = TurnQualityCache.get_instance()
    tq = await cache.aread(turn_id)

    if tq is None:
        raise HTTPException(status_code=404, detail="Turn quality not found")

    return tq


class PinChatRequest(BaseModel):
    pin: bool
    user_id: str


@router.patch("/chats/{chat_id}/pin", response_model=Chat)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
async def pin_chat(
    chat_id: UUID,
    pin_request: PinChatRequest,
) -> Chat:
    """
    Pin or unpin a chat for a user.
    """
    async with AsyncSession(get_async_engine()) as session:
        async with session.begin():  # Use a transaction
            # Get the chat and verify ownership
            try:
                stmt = select(Chat).where(Chat.chat_id == chat_id)  # type: ignore[arg-type]
                result = await session.execute(stmt)
                chat = result.scalar_one_or_none()
                if not chat:
                    raise HTTPException(status_code=404, detail="Chat not found")
                if str(chat.creator_user_id) != pin_request.user_id:
                    raise HTTPException(status_code=403, detail="Not authorized to modify this chat")

                if pin_request.pin:
                    # Check current number of pinned chats
                    query = (
                        select(func.count())
                        .select_from(Chat)
                        .where(Chat.creator_user_id == pin_request.user_id)  # type: ignore[arg-type]
                        .where(Chat.deleted_at.is_(None))  # type: ignore
                        .where(Chat.is_pinned.is_(True))  # type: ignore
                    )
                    result = await session.execute(query)
                    pinned_count = result.scalar()

                    if pinned_count >= MAX_PINNED_CHATS:  # type: ignore
                        raise HTTPException(status_code=400, detail=f"Cannot pin more than {MAX_PINNED_CHATS} chats")

                # Update pin status
                chat.is_pinned = pin_request.pin
                await session.flush()

                # Create a detached copy before closing the session
                chat_dict = {
                    "chat_id": chat.chat_id,
                    "creator_user_id": chat.creator_user_id,
                    "created_at": chat.created_at,
                    "modified_at": chat.modified_at,
                    "deleted_at": chat.deleted_at,
                    "is_pinned": chat.is_pinned,
                    "title": chat.title,
                    "path": chat.path,
                    "is_public": chat.is_public,
                }

                return Chat(**chat_dict)

            except HTTPException as he:
                raise HTTPException(status_code=he.status_code, detail=he.detail) from he
            except Exception as e:
                log_dict = {
                    "message": "Failed to pin/unpin chat",
                    "chat_id": str(chat_id),
                    "user_id": pin_request.user_id,
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                raise HTTPException(status_code=500, detail="Failed to pin/unpin chat") from e


@router.get("/chats/pinned", response_model=list[Chat])
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
async def get_user_pinned_chats(
    user_id: Annotated[str, Query(title="The ID of the user", required=True)],
) -> list[Chat]:
    """
    Get pinned chats for a user.

    Args:
        user_id (UUID): The UUID of the user

    Returns:
        list[Chat]: List of pinned Chat objects, ordered by creation date descending
    """
    try:
        async with AsyncSession(get_async_engine()) as session:
            query = (
                select(Chat)
                .where(Chat.creator_user_id == user_id)  # type: ignore[arg-type]
                .where(Chat.deleted_at.is_(None))  # type: ignore
                .where(Chat.is_pinned.is_(True))  # type: ignore
                .order_by(Chat.created_at.desc())  # type: ignore
                .limit(MAX_PINNED_CHATS)
            )

            result = await session.execute(query)
            chats = result.scalars().all()
            session.expunge_all()
            return list(chats)

    except Exception as e:
        log_dict = {
            "message": "Failed to fetch pinned chats",
            "user_id": str(user_id),
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail="Failed to fetch pinned chats") from e


class ChatResponse(BaseModel):
    """Response model for chat listing."""

    chats: list[Chat]
    has_more: bool


class GetChatsParams(BaseModel):
    """Parameters for fetching chat history."""

    user_id: Annotated[str, Query(title="The ID of the user", required=True)]
    last_chat_time: Annotated[
        datetime | None,
        Query(
            title="Timestamp of the last chat seen",
            description="Must be provided together with last_chat_id",
            default=None,
        ),
    ]
    last_chat_id: Annotated[
        UUID | None,
        Query(
            title="ID of the last chat seen", description="Must be provided together with last_chat_time", default=None
        ),
    ]
    page_size: Annotated[
        int,
        Query(title="Number of records to fetch", description="Number of chats to return per page", default=50, ge=1),
    ]

    @validator("user_id")
    def validate_user_id(cls, v: str) -> str:
        """Ensure user_id is not empty."""
        if not v or not v.strip():
            raise ValueError("user_id is required and cannot be empty")
        return v.strip()

    @validator("last_chat_id")
    def validate_last_chat_pair(cls, v: UUID | None, values: dict) -> UUID | None:
        """Ensure both chat values are provided if one is provided."""
        has_time = values.get("last_chat_time") is not None
        has_id = v is not None

        if has_time != has_id:
            raise ValueError(
                "Invalid pagination parameters: "
                "last_chat_time and last_chat_id must either both be provided or both be None. "
                f"Received: last_chat_time={'present' if has_time else 'missing'}, "
                f"last_chat_id={'present' if has_id else 'missing'}"
            )
        return v


@router.get("/chats", response_model=ChatResponse)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
async def get_user_chats(params: Annotated[GetChatsParams, Depends()]) -> ChatResponse:
    """
    Get non-deleted chats for a user with pagination for infinite scroll.

    If no last chat time or id is provided, returns the most recent chats.

    Args:
        params: Validated parameters for fetching chats.

    Returns:
        ChatResponse: List of chats and pagination info.

    Raises:
        HTTPException: If there's an error fetching chats.
    """
    try:
        async with AsyncSession(get_async_engine()) as session:
            query = (
                select(Chat)
                .where(Chat.creator_user_id == params.user_id)  # type: ignore[arg-type]
                .where(Chat.deleted_at.is_(None))  # type: ignore
            )

            # Only apply filtering if both parameters are provided
            if params.last_chat_time and params.last_chat_id:
                query = query.where(
                    (Chat.created_at, Chat.chat_id) < (params.last_chat_time, params.last_chat_id)  # type: ignore
                )

            query = query.order_by(Chat.created_at.desc(), Chat.chat_id.desc()).limit(params.page_size + 1)  # type: ignore

            result = await session.execute(query)
            chats = result.scalars().all()
            session.expunge_all()

            has_more = len(chats) > params.page_size
            result_chats = list(chats[: params.page_size])

            return ChatResponse(chats=result_chats, has_more=has_more)

    except Exception as e:
        log_dict = {
            "message": "Failed to fetch chat history",
            "params": params.dict(),
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail={"message": "Failed to fetch chat history", "error": str(e)}) from e


class MessageDebugInfo(BaseModel):
    message_id: UUID
    language_code: str
    modifiers: list[tuple[str, str]]


@router.get("/chat_messages/{message_id}/debug_info", response_model=MessageDebugInfo | None)
async def get_message_debug_info(message_id: UUID) -> MessageDebugInfo | None:
    try:
        async with AsyncSession(get_async_engine()) as session:
            stmt = (
                select(ChatMessage)
                .options(selectinload(ChatMessage.prompt_modifiers))  # type: ignore
                .where(ChatMessage.message_id == message_id)  # type: ignore
            )
            result = await session.execute(stmt)
            message = result.scalar_one_or_none()

            if not message:
                return None

            modifiers = [(mod.name, mod.text) for mod in message.prompt_modifiers]
            language_code = str(message.language_code) if message.language_code else ""

            return MessageDebugInfo(
                message_id=message_id,
                language_code=language_code,
                modifiers=modifiers,
            )

    except Exception as e:
        log_dict = {
            "message": f"Error getting message debug info: {str(e)}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
