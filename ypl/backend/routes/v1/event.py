from typing import Annotated

from fastapi import APIRouter, Query, status
from fastapi.responses import JSONResponse

from ypl.backend.event.event import (
    CreateEventRequest,
    EventsResponse,
    create_new_event,
    get_user_events,
)
from ypl.backend.utils.async_utils import create_background_task

router = APIRouter(prefix="/admin")


@router.get("/events")
async def get_events(
    user_id: Annotated[str | None, Query(description="Optional User ID to filter events")] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Number of events to return")] = 50,
    offset: Annotated[int, Query(ge=0, description="Number of events to skip")] = 0,
) -> EventsResponse:
    """Get events with pagination support.

    Args:
        user_id: Optional User ID to filter events for
        limit: Maximum number of events to return (default: 50, max: 100)
        offset: Number of events to skip for pagination (default: 0)

    Returns:
        EventsResponse containing the list of events and pagination info
    """
    return await get_user_events(user_id=user_id, limit=limit, offset=offset)


@router.post("/events", status_code=status.HTTP_202_ACCEPTED)
async def create_event(
    request: CreateEventRequest,
) -> JSONResponse:
    """Create a new event asynchronously.

    Args:
        request: Event details including user_id, event_name, event_category, and optional parameters

    Returns:
        JSONResponse with acknowledgment that event creation has been initiated
    """
    # Start event creation in background
    create_background_task(create_new_event(request))

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"message": "Event creation initiated", "user_id": request.user_id},
    )
