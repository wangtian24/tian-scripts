import asyncio
import logging
from collections.abc import Mapping, Sequence, Set
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy.orm import joinedload, selectinload
from sqlmodel import desc, select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.abuse import AbuseException
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack_with_user_name_bg
from ypl.backend.utils.ip_utils import get_ip_details
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import haversine_distance
from ypl.db.abuse import AbuseActionType, AbuseEvent, AbuseEventState, AbuseEventType
from ypl.db.events import Event
from ypl.db.invite_codes import SpecialInviteCode, SpecialInviteCodeClaimLog
from ypl.db.users import IPs, User


async def get_referring_user(session: AsyncSession, user_id: str) -> User | None:
    """Get the user who referred the user `user_id`, if any."""
    query = (
        select(User)
        .join(SpecialInviteCode)
        .join(SpecialInviteCodeClaimLog)
        .options(joinedload(User.ip_details))  # type: ignore
        .where(SpecialInviteCodeClaimLog.user_id == user_id)
    )

    return (await session.exec(query)).first()


async def get_referred_users(session: AsyncSession, user_id: str) -> Sequence[User]:
    """Get all users who were referred by the user `user_id`."""
    query = (
        select(User)
        .join(SpecialInviteCodeClaimLog)
        .join(SpecialInviteCode)
        .options(joinedload(User.ip_details))  # type: ignore
        .where(SpecialInviteCode.creator_user_id == user_id)
    )
    return (await session.exec(query)).unique().all()


async def get_recent_users(session: AsyncSession, min_creation_time: datetime) -> Sequence[User]:
    query = (
        select(User)
        .options(joinedload(User.ip_details))  # type: ignore
        .where(User.created_at > min_creation_time, User.deleted_at.is_(None))  # type: ignore
    )
    return (await session.exec(query)).unique().all()


def ip_details_str(ip_details: list[IPs] | None) -> str:
    ips = [ip.ip for ip in ip_details] if ip_details else []
    return ", ".join(sorted(ips))


async def create_abuse_event(
    session: AsyncSession,
    user: User,
    event_type: AbuseEventType,
    event_details: dict[str, Any],
    actions: Set[AbuseActionType] = frozenset(),
    skip_if_same_event_within_time_window: timedelta | None = timedelta(hours=1),
) -> None:
    if user.is_internal() and settings.ENVIRONMENT == "production":
        logging.info(f"Skipping abuse event {event_type} for internal user {user.email}")
        return

    if skip_if_same_event_within_time_window:
        query = select(AbuseEvent.abuse_event_id).where(
            AbuseEvent.user_id == user.user_id,
            AbuseEvent.event_type == event_type,
            AbuseEvent.created_at > datetime.now() - skip_if_same_event_within_time_window,  # type: ignore
        )
        result = await session.exec(query)
        if result.first():
            logging.info(
                f"Skipping abuse event {event_type} for user {user.user_id} because an event of this type already "
                f"exists within the {skip_if_same_event_within_time_window} time window"
            )
            return

    event = AbuseEvent(user_id=user.user_id, event_type=event_type, event_details=event_details)

    if AbuseActionType.SLACK_REPORT in actions:
        log_dict = {
            "message": f":red_circle: Abuse event detected: {event_type.value}",
            "user_id": user.user_id,
            "event_type": event_type.value,
            "event_details": event_details,
        }
        post_to_slack_with_user_name_bg(user.user_id, json_dumps(log_dict), settings.SLACK_WEBHOOK_CASHOUT)

    logging.warning(
        json_dumps(
            {
                "message": "abuse event created",
                "user_id": user.user_id,
                "event_type": event_type,
                "event_details": event_details,
            },
            indent=2,
        )
    )
    session.add(event)
    await session.commit()

    if AbuseActionType.RAISE_EXCEPTION in actions:
        raise AbuseException(f"Abuse event {event_type} for user {user.user_id}")


async def get_abuse_events(
    user_id: str | None = None,
    event_type: AbuseEventType | None = None,
    state: AbuseEventState | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[Sequence[AbuseEvent], bool]:
    async with get_async_session() as session:
        query = select(AbuseEvent).options(selectinload(AbuseEvent.user)).order_by(AbuseEvent.created_at.desc())  # type: ignore
        if user_id:
            query = query.where(AbuseEvent.user_id == user_id)
        if event_type:
            query = query.where(AbuseEvent.event_type == event_type)
        if state:
            query = query.where(AbuseEvent.state == state)
        query = query.offset(offset).limit(limit + 1)
        results = await session.exec(query)
        events = results.all()
        has_more_rows = len(events) > limit
        if has_more_rows:
            events = events[:-1]
        return events, has_more_rows


async def review_abuse_event(abuse_event_id: UUID, reviewer: str, notes: str | None = None) -> AbuseEvent | None:
    async with get_async_session() as session:
        query = (
            select(AbuseEvent).options(joinedload(AbuseEvent.user)).where(AbuseEvent.abuse_event_id == abuse_event_id)  # type: ignore
        )
        result = await session.execute(query)
        event: AbuseEvent | None = result.scalar_one_or_none()

        if event:
            event.state = AbuseEventState.REVIEWED
            event.reviewed_at = datetime.now()
            event.reviewed_by = reviewer
            event.review_notes = notes
            await session.commit()

        return event


async def get_user_events(
    user_id: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    event_names: list[str] | None = None,
    limit: int = 50,
) -> Sequence[tuple[str, str, Mapping[Any, Any] | None, datetime | None]]:
    """Returns tuples of (name, category, params, created_at) for the last `limit` events of the user."""
    async with get_async_session() as session:
        query = (
            select(Event.event_name, Event.event_category, Event.event_params, Event.created_at)
            .where(Event.user_id == user_id)
            .order_by(desc(Event.created_at))
            .limit(limit)
        )
        if start_time:
            query = query.where(Event.created_at > start_time)  # type: ignore
        if end_time:
            query = query.where(Event.created_at < end_time)  # type: ignore
        if event_names:
            query = query.where(Event.event_name.in_(event_names))  # type: ignore
        return (await session.exec(query)).all()


def is_impossible_travel_for_time_period(distance_km: float, time_difference_seconds: float) -> bool:
    """Check if travel is impossible based on distance and time difference.

    Uses different thresholds based on time periods, considering commercial aircraft speeds:
    - Within 30 minutes: max 100km
    - Within 1 hour: max 900km
    - Within 2 hours: max 1800km
    - Within 4 hours: max 3600km
    - Within 8 hours: max 7200km
    - Within 12 hours: max 10800km
    """
    time_hours = time_difference_seconds / 3600

    thresholds = [
        (0.5, 100),
        (1, 900),
        (2, 1800),
        (4, 3600),
        (8, 7200),
        (12, 10800),
    ]

    for max_hours, max_distance in thresholds:
        if time_hours <= max_hours:
            return distance_km > max_distance

    return distance_km > 10800


async def is_impossible_travel(user_id: str, current_ip: str) -> bool:
    """Check if the current IP location indicates impossible travel from previous locations.

    Args:
        user_id: The user ID to check
        current_ip: The current IP address

    Returns:
        True if the travel is impossible (indicating potential abuse), False otherwise
    """
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(hours=12)

    current_ip_info, events = await asyncio.gather(
        get_ip_details(current_ip),
        get_user_events(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        ),
    )

    if not current_ip_info or not current_ip_info.get("loc"):
        return False

    current_loc = current_ip_info["loc"]
    if not isinstance(current_loc, str):
        return False

    try:
        current_lat, current_lon = map(float, current_loc.split(","))
    except (ValueError, AttributeError):
        # incase of an error, return false to avoid bad user experience
        logging.warning(f"Invalid location for IP {current_ip} with user_id {user_id}: {current_loc}")
        return False

    unique_ips = set()
    ip_times = {}
    for _, _, event_params, created_at in events:
        if event_params and "ip" in event_params:
            ip = event_params["ip"]
            unique_ips.add(ip)
            if created_at:
                ip_times[ip] = created_at.replace(tzinfo=UTC)

    if not unique_ips:
        return False

    for prev_ip in unique_ips:
        if prev_ip == current_ip:
            continue

        prev_ip_info = await get_ip_details(prev_ip)
        if not prev_ip_info or not prev_ip_info.get("loc"):
            continue

        prev_loc = prev_ip_info["loc"]
        if not prev_loc:
            continue

        try:
            prev_lat, prev_lon = map(float, prev_loc.split(","))
        except (ValueError, AttributeError):
            continue

        distance = haversine_distance(prev_lat, prev_lon, current_lat, current_lon)
        prev_time = ip_times[prev_ip]
        if prev_time:
            time_diff = (end_time - prev_time).total_seconds()
            if is_impossible_travel_for_time_period(distance, time_diff):
                async with get_async_session() as session:
                    user = await session.get(User, user_id)
                    if user:
                        await create_abuse_event(
                            session=session,
                            user=user,
                            event_type=AbuseEventType.IMPOSSIBLE_TRAVEL,
                            event_details={
                                "current_ip": current_ip,
                                "current_location": f"{current_lat}, {current_lon}",
                                "previous_ip": prev_ip,
                                "previous_location": f"{prev_lat}, {prev_lon}",
                                "distance_km": distance,
                                "time_difference_seconds": time_diff,
                            },
                            actions={AbuseActionType.SLACK_REPORT},
                        )
                return True

    return False
