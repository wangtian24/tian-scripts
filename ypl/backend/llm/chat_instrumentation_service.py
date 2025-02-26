import asyncio
from enum import Enum
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel

from ypl.backend.db import get_async_session
from ypl.db.chats import ChatInstrumentation


class EventSource(Enum):
    HEAD_CLIENT = "head_client"
    HEAD_SERVER = "head_server"
    MIND_SERVER = "mind_server"


class ChatInstrumentationRequest(BaseModel):
    message_id: UUID
    event_source: EventSource
    streaming_metrics: dict[str, Any]


class ChatInstrumentationResponse(BaseModel):
    message_id: UUID
    head_client: dict[str, Any] | None
    head_server: dict[str, Any] | None
    mind_server: dict[str, Any] | None
    workflow_status: dict[str, Any] | None
    analysis_output: dict[str, Any] | None

    class Config:
        from_attributes = True  # This enables ORM model -> Pydantic conversion


async def create_or_update_instrumentation(
    chat_instrumentation_request: ChatInstrumentationRequest,
) -> None:
    """
    Creates or updates chat instrumentation data.

    Args:
        message_id: UUID of the message
        event_source: Source of the event (head_client, head_server, mind_server)
        streaming_metrics: Metrics data for the chat
        session: Database session

    Returns:
        Dict containing status and message

    Raises:
        HTTPException: If event_source is invalid
    """
    # Validate event source
    if chat_instrumentation_request.event_source not in EventSource:
        raise HTTPException(status_code=400, detail=f"Invalid event_source. Must be one of: {', '.join(EventSource)}")
    async with get_async_session() as session:
        # Get existing record or create new one
        instrumentation = await session.get(ChatInstrumentation, chat_instrumentation_request.message_id)
        if not instrumentation:
            instrumentation = ChatInstrumentation(message_id=chat_instrumentation_request.message_id)
            session.add(instrumentation)

        # Update the appropriate column based on event_source
        match chat_instrumentation_request.event_source:
            case EventSource.HEAD_CLIENT:
                instrumentation.head_client = chat_instrumentation_request.streaming_metrics
            case EventSource.HEAD_SERVER:
                instrumentation.head_server = chat_instrumentation_request.streaming_metrics
            case EventSource.MIND_SERVER:
                instrumentation.mind_server = chat_instrumentation_request.streaming_metrics
            case _:
                raise ValueError(f"Unexpected event source: {chat_instrumentation_request.event_source}")

        await session.commit()


async def get_instrumentation(message_id: UUID) -> ChatInstrumentationResponse:
    """Get chat instrumentation data for a message ID.

    Args:
        message_id: UUID of the message to get instrumentation for

    Returns:
        Dict containing the instrumentation data

    Raises:
        HTTPException: If instrumentation not found for message_id
    """
    async with get_async_session() as session:
        instrumentation = await session.get(ChatInstrumentation, message_id)
        if not instrumentation:
            raise HTTPException(status_code=404, detail=f"No instrumentation found for message_id: {message_id}")

        return ChatInstrumentationResponse.model_validate(instrumentation)


async def create_or_update_instrumentation_batch(batch_request: list[ChatInstrumentationRequest]) -> None:
    """Create or update instrumentation entries for multiple messages in batch.

    Args:
        requests: List of dicts containing message_id, event_source, and streaming_metrics
    """
    await asyncio.gather(*(create_or_update_instrumentation(req) for req in batch_request))
