from typing import Annotated

from fastapi import APIRouter, Query

from ypl.backend.message.message import MessagesResponse, get_user_messages

router = APIRouter()


@router.get("/messages")
async def get_messages(
    user_id: Annotated[str | None, Query(description="Optional User ID to filter messages")] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Number of messages to return")] = 50,
    offset: Annotated[int, Query(ge=0, description="Number of messages to skip")] = 0,
) -> MessagesResponse:
    """Get messages with pagination support.

    Args:
        user_id: Optional User ID to filter messages for
        limit: Maximum number of messages to return (default: 50, max: 100)
        offset: Number of messages to skip for pagination (default: 0)

    Returns:
        MessagesResponse containing the list of messages and pagination info
    """
    return await get_user_messages(user_id=user_id, limit=limit, offset=offset)
