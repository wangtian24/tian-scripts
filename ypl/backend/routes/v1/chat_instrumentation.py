from asyncio import create_task
from uuid import UUID

from fastapi import APIRouter, Response

from ypl.backend.llm.chat_instrumentation_service import (
    ChatInstrumentationRequest,
    ChatInstrumentationResponse,
    create_or_update_instrumentation,
    create_or_update_instrumentation_batch,
    get_instrumentation,
)

router = APIRouter()


@router.post("/chat/instrumentation")
async def create_chat_instrumentation(
    request: ChatInstrumentationRequest | list[ChatInstrumentationRequest],
) -> Response:
    """Create instrumentation entries for single or multiple messages.

    Args:
        request: Single instrumentation request or list of requests

    Returns:
        Dict indicating success and number of tasks queued
    """
    if isinstance(request, list):
        create_task(create_or_update_instrumentation_batch(request))
    else:
        create_task(create_or_update_instrumentation(request))
    return Response(status_code=204)


@router.get("/chat/instrumentation/{message_id}")
async def get_chat_instrumentation(message_id: UUID) -> ChatInstrumentationResponse:
    """Get chat instrumentation data for a message ID.

    Args:
        message_id: UUID of the message to get instrumentation for

    Returns:
        Dict containing the instrumentation data

    Raises:
        HTTPException: If instrumentation not found for message_id
    """
    return await get_instrumentation(message_id)
