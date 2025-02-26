import logging
from collections.abc import Sequence, Set
from typing import Any

from sqlalchemy.orm import selectinload
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.abuse import AbuseException
from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.abuse import AbuseActionType, AbuseEvent, AbuseEventState, AbuseEventType
from ypl.db.invite_codes import SpecialInviteCode, SpecialInviteCodeClaimLog
from ypl.db.users import User


async def get_referring_user(session: AsyncSession, user_id: str) -> User | None:
    """Get the user who referred the user `user_id`, if any."""
    query = (
        select(User)
        .join(SpecialInviteCode)
        .join(SpecialInviteCodeClaimLog)
        .where(SpecialInviteCodeClaimLog.user_id == user_id)
    )

    return (await session.exec(query)).first()


async def get_referred_users(session: AsyncSession, user_id: str) -> Sequence[User]:
    """Get all users who were referred by the user `user_id`."""
    query = (
        select(User)
        .join(SpecialInviteCodeClaimLog)
        .join(SpecialInviteCode)
        .where(SpecialInviteCode.creator_user_id == user_id)
    )
    return (await session.exec(query)).all()


async def create_abuse_event(
    session: AsyncSession,
    user: User,
    event_type: AbuseEventType,
    event_details: dict[str, Any],
    actions: Set[AbuseActionType] = frozenset(),
) -> None:
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
