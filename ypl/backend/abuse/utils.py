import logging
from collections.abc import Sequence, Set
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy.orm import joinedload, selectinload
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.abuse import AbuseException
from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.abuse import AbuseActionType, AbuseEvent, AbuseEventState, AbuseEventType
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
    if user.is_internal():
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
        # TODO(gilad): add
        pass

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
