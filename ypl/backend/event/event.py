import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import desc, select
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack_with_user_name
from ypl.backend.utils.ip_utils import store_ip_details
from ypl.backend.utils.json import json_dumps
from ypl.db.events import Event
from ypl.db.redis import get_upstash_redis_client
from ypl.db.users import SYSTEM_USER_ID, User

EVENT_RATE_LIMIT_SKIP_CHECK_KEY = "event:rate_limit:skip_checking"
EVENT_RATE_LIMIT_WINDOW_SECONDS: Final[int] = 10

RATE_LIMITED_EVENTS = {
    "send_feedback": {"max_requests": 5, "window_seconds": 10},
    "start_chat": {"max_requests": 3, "window_seconds": 10},
    "initiated_cashout": {"max_requests": 3, "window_seconds": 10},
    "eval_qt": {"max_requests": 3, "window_seconds": 10},
}

SLACK_WEBHOOK_EVENT_RATE_LIMIT = settings.SLACK_WEBHOOK_URL


async def skip_event_rate_limit_check() -> bool:
    """Check if event rate limit checking should be skipped."""

    if settings.ENVIRONMENT != "production":
        return True

    redis = await get_upstash_redis_client()
    skip_checking = await redis.get(EVENT_RATE_LIMIT_SKIP_CHECK_KEY)
    if skip_checking or skip_checking is None:
        return True
    return False


async def check_event_rate_limit(user_id: str, event_name: str) -> None:
    """Check if the user has exceeded the rate limit for a specific event."""
    if event_name not in RATE_LIMITED_EVENTS or await skip_event_rate_limit_check():
        return

    event_config = RATE_LIMITED_EVENTS[event_name]
    max_requests = event_config["max_requests"]
    window_seconds = event_config["window_seconds"]

    redis = await get_upstash_redis_client()
    key = f"event:rate_limit:{event_name}:{user_id}"

    try:
        count_str = await redis.get(key)
        count = int(count_str) if count_str is not None else 0

        if count >= max_requests:
            log_dict = {
                "message": f":warning: Repetitive activity detected for event {event_name} by user {user_id}",
                "current_count": count,
                "max_requests": max_requests,
                "window_seconds": window_seconds,
            }
            logging.warning(json_dumps(log_dict))
            await post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_EVENT_RATE_LIMIT)

        await redis.incr(key)
        if count == 0:
            await redis.expire(key, window_seconds)

    except Exception as e:
        log_dict = {
            "message": "Error checking event rate limit",
            "user_id": user_id,
            "event_name": event_name,
            "error": str(e),
        }
        logging.warning(json_dumps(log_dict))
        return


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

    user_id: str | None = None
    event_name: str = "UNKNOWN"
    event_category: str = "UNKNOWN"
    event_params: dict | None = None
    event_guestivity_details: dict | None = None
    event_dedup_id: str | None = None


async def get_user_events(
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> EventsResponse:
    """Get events for a user with pagination support."""
    try:
        async with get_async_session() as session:
            query = select(Event).order_by(desc(Event.created_at))

            if user_id is not None:
                query = query.where(Event.user_id == user_id)

            query = query.offset(offset).limit(limit + 1)

            result = await session.execute(query)
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


async def create_new_event(request: CreateEventRequest) -> EventResponse | None:
    """Create a new event."""
    log_dict = {
        "message": "Creating event",
        "request": request,
    }
    logging.info(json_dumps(log_dict))
    try:
        #  check if user_id is blank in which case check if creator_user_email is present
        # and pull the user_id from the users table based on the creator_user_email
        if not request.user_id:
            #  check if creator_user_email is present in the event_params
            if request.event_params and request.event_params.get("creator_user_email"):
                async with get_async_session() as session:
                    users = await session.execute(
                        select(User).where(User.email == request.event_params["creator_user_email"])
                    )
                    user = users.scalar_one_or_none()
                    if user:
                        request.user_id = user.user_id

        if not request.user_id:
            request.user_id = SYSTEM_USER_ID

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

            # TODO: This is a temporary fix to update the country code in the users table
            #  if the country code is null in the users table for the user_id,
            #  then we need to update the country code in the users table from the event_params
            if event.event_params and event.event_params.get("country_code"):
                users = await session.execute(select(User).where(User.user_id == event.user_id))
                user = users.scalar_one_or_none()
                if user and user.country_code is None:
                    log_dict = {
                        "message": "Updating country code",
                        "user_id": event.user_id,
                        "country_code": event.event_params["country_code"],
                    }
                    logging.info(json_dumps(log_dict))
                    user.country_code = event.event_params["country_code"]
                    await session.commit()

            #  check potential bot activity
            await check_event_rate_limit(request.user_id, request.event_name)

            #  store ip details
            if event.event_params and event.event_params.get("ip"):
                await store_ip_details(event.event_params["ip"], request.user_id)

            #  send slack notification for admin actions
            if event.event_category == "ADMIN_ACTION":
                await send_slack_notification_for_admin_actions(event)

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

    except Exception as e:
        # Special handling for foreign key violations related to user_id
        # as sometime users are on waitlist and their user_id is not present in the users table
        if (
            isinstance(e, SQLAlchemyError)
            and "violates foreign key constraint" in str(e)
            and "fk_events_user_id_users" in str(e)
        ):
            log_dict = {
                "message": "Warning: Event creation failed due to non-existent user_id",
                "user_id": request.user_id,
                "event_name": request.event_name,
                "event_category": request.event_category,
                "error": str(e),
            }
            logging.warning(json_dumps(log_dict))
            return None

        log_dict = {
            "message": "Unexpected error creating event",
            "user_id": request.user_id,
            "event_name": request.event_name,
            "event_category": request.event_category,
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        return None


async def send_slack_notification_for_admin_actions(event: Event) -> None:
    # Currently only sending invite code events to the guest management slack channel.
    if event.event_name.endswith("INVITE_CODE"):
        slack_message = f"- {event.event_name}"
        if event.event_params:
            slack_message += f"\nPerformed by: {event.event_params.get('creator_user_email')}"
        slack_message += f"\nDetails:\n {json_dumps(event.event_params, indent=2)}"

        await post_to_slack_with_user_name(event.user_id, slack_message, settings.GUEST_MANAGEMENT_SLACK_WEBHOOK_URL)
