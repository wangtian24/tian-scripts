import logging
import uuid
from enum import Enum
from typing import TypedDict, cast

from sqlalchemy import func, or_
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from ypl.backend.db import get_async_engine
from ypl.db.users import (
    Capability,
    CapabilityStatus,
    User,
    UserCapabilityOverride,
    UserCapabilityStatus,
    WaitlistedUser,
)

UNKNOWN_USER = "Unknown User"
WAITLISTED_SUFFIX = " (**Waitlisted**)"


async def fetch_user_names(user_ids: list[str]) -> dict[str, str]:
    """Fetch multiple user names from the database in a single query."""
    try:
        engine = get_async_engine()
        async with AsyncSession(engine) as session:
            # First try to get names from users table
            query = select(User).where(
                User.user_id.in_(user_ids),  # type: ignore
            )
            users = (await session.exec(query)).all()

            name_dict = {user_id: user_id for user_id in user_ids}
            name_dict.update({user.user_id: str(user.name) for user in users if user.name})

            # For any remaining user_ids without names, check waitlisted_users table
            remaining_ids = [uid for uid in user_ids if name_dict[uid] == uid]
            if remaining_ids:
                waitlist_query = select(WaitlistedUser).where(
                    WaitlistedUser.waitlisted_user_id.in_([cast(uuid.UUID, uid) for uid in remaining_ids]),  # type: ignore
                )
                waitlisted_users = (await session.exec(waitlist_query)).all()

                # Update names from waitlisted users if found, adding the waitlisted suffix
                name_dict.update(
                    {
                        str(waitlisted_user.waitlisted_user_id): WAITLISTED_SUFFIX + str(waitlisted_user.name)
                        for waitlisted_user in waitlisted_users
                        if waitlisted_user.name
                    }
                )

            return name_dict

    except Exception as e:
        logging.exception(f"Failed to fetch users from database: {e}")
        return {user_id: user_id for user_id in user_ids}


async def fetch_user_name(user_id: str) -> str:
    """Fetch a single user name from the database."""
    try:
        engine = get_async_engine()
        async with AsyncSession(engine) as session:
            # First try users table
            query = select(User).where(
                User.user_id == user_id,
            )
            user = (await session.exec(query)).first()

            if user and user.name:
                return str(user.name)

            # If not found or no name, check waitlisted_users table
            waitlist_query = select(WaitlistedUser).where(
                WaitlistedUser.waitlisted_user_id == cast(uuid.UUID, user_id),
            )
            waitlisted_user = (await session.exec(waitlist_query)).first()

            if waitlisted_user and waitlisted_user.name:
                return WAITLISTED_SUFFIX + str(waitlisted_user.name)

            return UNKNOWN_USER

    except Exception as e:
        logging.exception(f"Failed to fetch user name from database: {e}")
        return UNKNOWN_USER


class CapabilityType(str, Enum):
    """Enum representing different capability types in the system."""

    CASHOUT = "cashout"


class CapabilityOverrideDetails(TypedDict):
    status: UserCapabilityStatus
    override_config: dict | None


async def get_capability_override_details(
    session: AsyncSession, user_id: str, capability_name: CapabilityType
) -> CapabilityOverrideDetails | None:
    """Check if a user has an entry in the user capability override table for the given capability type.
    If no entry exists, return None.
    If an entry exists and is disabled, indicate that the user is not eligible for this capability.
    If enabled, return the override configurations.
    """
    now = func.now()

    capability_stmt = select(Capability).where(
        Capability.capability_name == capability_name.value,
        Capability.deleted_at.is_(None),  # type: ignore
        Capability.status == CapabilityStatus.ACTIVE,
    )
    capability = (await session.exec(capability_stmt)).first()

    if not capability:
        return None

    # Then check for any active overrides
    override_stmt = select(UserCapabilityOverride).where(
        UserCapabilityOverride.user_id == user_id,
        UserCapabilityOverride.capability_id == capability.capability_id,
        UserCapabilityOverride.deleted_at.is_(None),  # type: ignore
        or_(
            UserCapabilityOverride.effective_start_date.is_(None),  # type: ignore
            UserCapabilityOverride.effective_start_date <= now,
        ),
        or_(
            UserCapabilityOverride.effective_end_date.is_(None),  # type: ignore
            UserCapabilityOverride.effective_end_date > now,
        ),
    )
    override = (await session.exec(override_stmt)).first()

    if not override:
        return None

    return {"status": override.status, "override_config": override.override_config}
