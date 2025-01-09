import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ypl.backend.db import get_async_engine
from ypl.db.users import User, WaitlistedUser

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
            result = await session.execute(query)
            users = result.scalars().all()

            name_dict = {user_id: user_id for user_id in user_ids}
            name_dict.update({user.user_id: str(user.name) for user in users if user.name})

            # For any remaining user_ids without names, check waitlisted_users table
            remaining_ids = [uid for uid in user_ids if name_dict[uid] == uid]
            if remaining_ids:
                waitlist_query = select(WaitlistedUser).where(
                    WaitlistedUser.waitlisted_user_id.in_(remaining_ids),  # type: ignore
                )
                waitlist_result = await session.execute(waitlist_query)
                waitlisted_users = waitlist_result.scalars().all()

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
                User.user_id == user_id,  # type: ignore
            )
            result = await session.execute(query)
            user = result.scalar_one_or_none()

            if user and user.name:
                return str(user.name)

            # If not found or no name, check waitlisted_users table
            waitlist_query = select(WaitlistedUser).where(
                WaitlistedUser.waitlisted_user_id == user_id,  # type: ignore
            )
            waitlist_result = await session.execute(waitlist_query)
            waitlisted_user = waitlist_result.scalar_one_or_none()

            if waitlisted_user and waitlisted_user.name:
                return WAITLISTED_SUFFIX + str(waitlisted_user.name)

            return UNKNOWN_USER

    except Exception as e:
        logging.exception(f"Failed to fetch user name from database: {e}")
        return UNKNOWN_USER
