import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import desc, select

from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.events import Event

router = APIRouter(prefix="/admin", tags=["admin"])


@dataclass
class EventResponse:
    """Response model for a single event."""

    event_id: str
    user_id: str
    event_name: str
    event_category: str
    event_params: dict | None
    event_guestivity_details: dict | None
    event_dedup_id: str | None
    created_at: datetime | None


@dataclass
class EventsResponse:
    """Response model for a paginated list of events."""

    events: list[EventResponse]
    has_more_rows: bool


@dataclass
class CreateEventRequest:
    """Request model for creating a new event."""

    user_id: str
    event_name: str
    event_category: str = "UNKNOWN"
    event_params: dict | None = None
    event_guestivity_details: dict | None = None
    event_dedup_id: str | None = None


@router.get("/events")
async def get_events(
    user_id: Annotated[str, Query(..., description="User ID to get events for")],
    limit: Annotated[int, Query(ge=1, le=100, description="Number of events to return")] = 50,
    offset: Annotated[int, Query(ge=0, description="Number of events to skip")] = 0,
) -> EventsResponse:
    """Get events for a user with pagination support.

    Args:
        user_id: The ID of the user to get events for
        limit: Maximum number of events to return (default: 50, max: 100)
        offset: Number of events to skip for pagination (default: 0)

    Returns:
        EventsResponse containing the list of events and pagination info

    Raises:
        HTTPException: If there's an error fetching events from the database
    """
    try:
        async with get_async_session() as session:
            stmt = (
                select(Event)
                .where(Event.user_id == user_id)
                .order_by(desc(Event.created_at))
                .offset(offset)
                .limit(limit + 1)
            )

            result = await session.execute(stmt)
            events = result.scalars().all()

            has_more_rows = len(events) > limit
            if has_more_rows:
                events = events[:-1]

            log_dict = {
                "message": "Events found",
                "user_id": user_id,
                "events_count": len(events),
                "limit": limit,
                "offset": offset,
                "has_more_rows": has_more_rows,
            }
            logging.info(json_dumps(log_dict))

            return EventsResponse(
                events=[
                    EventResponse(
                        event_id=str(event.event_id),
                        user_id=event.user_id,
                        event_name=event.event_name,
                        event_category=event.event_category,
                        event_params=event.event_params,
                        event_guestivity_details=event.event_guestivity_details,
                        event_dedup_id=event.event_dedup_id,
                        created_at=event.created_at,
                    )
                    for event in events
                ],
                has_more_rows=has_more_rows,
            )

    except SQLAlchemyError as e:
        log_dict = {
            "message": "Database error getting events",
            "user_id": user_id,
            "limit": limit,
            "offset": offset,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch events from database",
        ) from e
    except Exception as e:
        log_dict = {
            "message": "Unexpected error getting events",
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


@router.post("/events", status_code=status.HTTP_201_CREATED)
async def create_event(
    request: CreateEventRequest,
) -> EventResponse:
    """Create a new event.

    Args:
        request: Event details including user_id, event_name, event_category, and optional parameters

    Returns:
        EventResponse containing the created event details

    Raises:
        HTTPException: If there's an error creating the event in the database
    """
    try:
        async with get_async_session() as session:
            event = Event(
                user_id=request.user_id,
                event_name=request.event_name,
                event_category=request.event_category,
                event_params=request.event_params,
                event_guestivity_details=request.event_guestivity_details,
                event_dedup_id=request.event_dedup_id,
            )
            session.add(event)
            await session.commit()

            log_dict = {
                "message": "Event created",
                "event_id": str(event.event_id),
                "user_id": event.user_id,
                "event_name": event.event_name,
                "event_category": event.event_category,
            }
            logging.info(json_dumps(log_dict))

            return EventResponse(
                event_id=str(event.event_id),
                user_id=event.user_id,
                event_name=event.event_name,
                event_category=event.event_category,
                event_params=event.event_params,
                event_guestivity_details=event.event_guestivity_details,
                event_dedup_id=event.event_dedup_id,
                created_at=event.created_at,
            )

    except SQLAlchemyError as e:
        log_dict = {
            "message": "Database error creating event",
            "user_id": request.user_id,
            "event_name": request.event_name,
            "event_category": request.event_category,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create event in database",
        ) from e
    except Exception as e:
        log_dict = {
            "message": "Unexpected error creating event",
            "user_id": request.user_id,
            "event_name": request.event_name,
            "event_category": request.event_category,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from e
