import asyncio
import json
import logging
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query

# from google.cloud import storage
from pydantic import BaseModel, validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from tenacity import retry, stop_after_attempt, wait_exponential

from ypl.backend.db import get_async_engine, get_async_session
from ypl.backend.llm.chat import (
    PromptModifierInfo,
    QuickTakeRequest,
    QuickTakeResponse,
    generate_quicktake,
    get_active_prompt_modifiers,
)
from ypl.backend.llm.search import search_chat_messages, search_chats
from ypl.backend.llm.turn_quality import label_turn_quality
from ypl.backend.rw_cache import TurnQualityCache
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import Chat, ChatMessage, MessageType, SuggestedTurnPrompt, SuggestedUserPrompt, Turn, TurnQuality

# Maximum number of pinned chats for a user.
MAX_PINNED_CHATS = 10

router = APIRouter()


@router.post("/chats/{chat_id}/generate_quicktake", response_model=QuickTakeResponse)
async def generate_quicktake_chat_id(
    request: QuickTakeRequest,
    chat_id: str = Path(..., description="The ID of the chat"),
) -> QuickTakeResponse:
    try:
        request.chat_id = chat_id if chat_id else request.chat_id
        return await generate_quicktake(request)
    except Exception as e:
        logging.exception(f"Error generating quicktake for chat {str(request)}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/chats/{chat_id}/turns/{turn_id}/generate_quicktake", response_model=QuickTakeResponse)
async def generate_quicktake_turn_id(
    request: QuickTakeRequest,
    chat_id: str = Path(..., description="The ID of the chat"),
    turn_id: str = Path(..., description="The ID of the turn"),
) -> QuickTakeResponse:
    try:
        request.chat_id = chat_id if chat_id else request.chat_id
        request.turn_id = turn_id if turn_id else request.turn_id
        return await generate_quicktake(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/chats/{chat_id}/turns/{turn_id}:label_quality", response_model=TurnQuality)
async def label_quality(chat_id: UUID, turn_id: UUID) -> TurnQuality:
    try:
        # TODO(gilad): Now that the backend handles streaming, it also initiates quality labeling directly, and the
        # client no longer needs to call this endpoint.
        # Until the call to this endpoint from the client is removed, add a short delay here (the client is not waiting
        # for a response), so that the turn quality is more likely to be cached by the time the client makes the call.
        # Once the client no longer calls this endpoint, remove this endpoint entirely.
        await asyncio.sleep(5)
        return await label_turn_quality(turn_id, chat_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


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


class ChatMessagesResponse(BaseModel):
    """Response model for chat messages listing."""

    messages: list[ChatMessage]
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
    annotations: dict[str, Any]
    metadata: dict[str, Any]
    prompt_difficulty: float | None
    prompt_difficulty_details: dict[str, Any] | None


@router.get("/chat_messages/{message_id}/debug_info", response_model=MessageDebugInfo | None)
async def get_message_debug_info(message_id: UUID) -> MessageDebugInfo | None:
    try:
        async with AsyncSession(get_async_engine()) as session:
            stmt = (
                select(ChatMessage)
                .options(selectinload(ChatMessage.prompt_modifiers))  # type: ignore
                .options(selectinload(ChatMessage.turn).selectinload(Turn.turn_quality))  # type: ignore
                .where(ChatMessage.message_id == message_id)  # type: ignore
            )
            result = await session.execute(stmt)
            message = result.scalar_one_or_none()

            if not message:
                return None

            modifiers = [(mod.name, mod.text) for mod in message.prompt_modifiers]
            language_code = str(message.language_code) if message.language_code else ""
            turn_quality = message.turn.turn_quality
            prompt_difficulty = turn_quality.prompt_difficulty if turn_quality else None
            prompt_difficulty_details = {"prompt_difficulty_details_raw_str": turn_quality.prompt_difficulty_details}
            if turn_quality.prompt_difficulty_details:
                try:
                    # Try to interpret it as a JSON object, fallback to raw string.
                    prompt_difficulty_details = json.loads(turn_quality.prompt_difficulty_details)
                except Exception:
                    pass  # Keep the raw string.

            return MessageDebugInfo(
                message_id=message_id,
                language_code=language_code,
                modifiers=modifiers,
                annotations=message.annotations or {},
                metadata=message.message_metadata or {},
                prompt_difficulty=prompt_difficulty,
                prompt_difficulty_details=prompt_difficulty_details,
            )

    except Exception as e:
        log_dict = {
            "message": f"Error getting message debug info: {str(e)}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/chats/prompt_modifiers", response_model=list[PromptModifierInfo])
async def get_prompt_modifiers() -> list[PromptModifierInfo]:
    try:
        return [
            PromptModifierInfo(prompt_modifier_id=str(m.prompt_modifier_id), name=m.name, description=m.description)
            for m in await get_active_prompt_modifiers()
        ]
    except Exception as e:
        log_dict = {"message": f"Error getting prompt modifiers: {str(e)}"}
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


class TurnAnnotationsResponse(BaseModel):
    comment: str | None = None
    positive_notes: list[tuple[str, str]] | None = None
    negative_notes: list[tuple[str, str]] | None = None


@router.get("/turns/{turn_id}/turn_annotations", response_model=TurnAnnotationsResponse)
async def get_turn_annotations(
    turn_id: str = Path(..., description="The ID of the turn"),
) -> TurnAnnotationsResponse:
    response = TurnAnnotationsResponse()
    try:
        async with get_async_session() as session:
            stmt = select(TurnQuality.prompt_difficulty_details).where(TurnQuality.turn_id == turn_id)  # type: ignore
            result = await session.execute(stmt)
            details = result.scalar_one_or_none()
            if details:
                details_dict = json.loads(details)
                response.comment = details_dict.get("comment")
                if "positive_notes" in details_dict:
                    response.positive_notes = [tuple(n.rsplit(maxsplit=1)) for n in details_dict["positive_notes"]]
                if "negative_notes" in details_dict:
                    response.negative_notes = [tuple(n.rsplit(maxsplit=1)) for n in details_dict["negative_notes"]]
    except Exception as e:
        log_dict = {"message": f"Error getting annotations for turn {turn_id}: {str(e)}"}
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
    return response


class SuggestedPrompt(BaseModel):
    prompt: str
    summary: str
    explanation: str | None = None


class SuggestedPromptsResponse(BaseModel):
    prompts: list[SuggestedPrompt]


@router.get("/turns/{turn_id}/suggested_followups", response_model=SuggestedPromptsResponse)
async def get_suggested_followups(turn_id: UUID) -> SuggestedPromptsResponse:
    try:
        async with get_async_session() as session:
            stmt = select(SuggestedTurnPrompt).where(
                SuggestedTurnPrompt.turn_id == turn_id,  # type: ignore
                SuggestedTurnPrompt.deleted_at.is_(None),  # type: ignore
            )
            result = await session.execute(stmt)
            suggested_followups = result.scalars().all()
            return SuggestedPromptsResponse(
                prompts=[SuggestedPrompt(prompt=sf.prompt, summary=sf.summary) for sf in suggested_followups]
            )
    except Exception as e:
        log_dict = {"message": f"Error getting suggested followups for turn {turn_id}: {str(e)}"}
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/users/{user_id}/conversation_starters", response_model=SuggestedPromptsResponse)
async def get_conversation_starters(user_id: str) -> SuggestedPromptsResponse:
    try:
        async with get_async_session() as session:
            stmt = select(SuggestedUserPrompt).where(
                SuggestedUserPrompt.user_id == user_id,  # type: ignore
                SuggestedUserPrompt.deleted_at.is_(None),  # type: ignore
            )
            result = await session.execute(stmt)
            conversation_starters = result.scalars().all()
            return SuggestedPromptsResponse(
                prompts=[
                    SuggestedPrompt(prompt=cs.prompt, summary=cs.summary, explanation=cs.explanation)
                    for cs in conversation_starters
                ]
            )
    except Exception as e:
        log_dict = {"message": f"Error getting conversation starters for user {user_id}: {str(e)}"}
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


class ChatSearchRequest(BaseModel):
    query: Annotated[str, Query(title="Search query", description="Text to search for in chats/messages")]
    limit: Annotated[
        int, Query(title="Result limit", description="Maximum number of results to return", ge=1, le=100)
    ] = 20
    offset: Annotated[int, Query(title="Result offset", description="Number of results to skip", ge=0)] = 0
    order_by: Annotated[
        Literal["relevance", "created_at"],
        Query(title="Sort order", description="How to order the results"),
    ] = "created_at"
    message_types: Annotated[
        tuple[MessageType, ...] | None,
        Query(title="Message types", description="Match only messages of these types"),
    ] = None
    creator_user_id: Annotated[
        str | None,
        Query(title="Creator user ID", description="Filter results to a specific user"),
    ] = None
    start_date: Annotated[
        datetime | None,
        Query(title="Start date", description="Filter results to chats/messages after this date"),
    ] = None
    end_date: Annotated[
        datetime | None,
        Query(title="End date", description="Filter results to chats/messages before this date"),
    ] = None
    message_fields: Annotated[
        tuple[str, ...] | None,
        Query(
            title="Message fields",
            description="Which fields to include in the response (the actual search is done on the message content)",
        ),
    ] = None


@router.get("/chats/search", response_model=ChatResponse)
async def chat_search(request: Annotated[ChatSearchRequest, Depends()]) -> ChatResponse:
    try:
        if len(request.query.strip()) == 0:
            raise ValueError("Query must not be empty")
        if request.limit < 1 or request.limit > 100:
            raise ValueError("Limit must be between 1 and 100")

        async with get_async_session() as session:
            results = await search_chats(
                session,
                request.query,
                limit=request.limit,
                offset=request.offset,
                message_types=request.message_types,
                creator_user_id=request.creator_user_id,
                start_date=request.start_date,
                end_date=request.end_date,
                order_by=request.order_by,
            )
        return ChatResponse(chats=results, has_more=len(results) >= request.limit)

    except Exception as e:
        log_dict = {
            "message": "Error searching chats",
            "query": request.query,
            "full_request": request.model_dump_json(),
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=f"Error searching chats: {str(e)}") from e


@router.get("/chat_messages/search", response_model=ChatMessagesResponse)
async def chat_messages_search(request: Annotated[ChatSearchRequest, Depends()]) -> ChatMessagesResponse:
    try:
        if len(request.query.strip()) == 0:
            raise ValueError("Query must not be empty")
        if request.limit < 1 or request.limit > 100:
            raise ValueError("Limit must be between 1 and 100")

        async with get_async_session() as session:
            results = await search_chat_messages(
                session,
                request.query,
                limit=request.limit,
                offset=request.offset,
                message_types=request.message_types,
                creator_user_id=request.creator_user_id,
                start_date=request.start_date,
                end_date=request.end_date,
                order_by=request.order_by,
                message_fields=request.message_fields,
            )
        return ChatMessagesResponse(messages=results, has_more=len(results) >= request.limit)

    except Exception as e:
        log_dict = {
            "message": "Error searching messages",
            "query": request.query,
            "full_request": request.model_dump_json(),
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=f"Error searching messages: {str(e)}") from e
