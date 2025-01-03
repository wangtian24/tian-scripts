import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ypl.backend.db import get_async_engine
from ypl.db.users import User

UNKNOWN_USER = "Unknown User"


async def fetch_user_names(user_ids: list[str]) -> dict[str, str]:
    """Fetch multiple user names from the database in a single query."""
    try:
        engine = get_async_engine()
        async with AsyncSession(engine) as session:
            query = select(User).where(
                User.user_id.in_(user_ids),  # type: ignore
            )
            result = await session.execute(query)
            users = result.scalars().all()

            name_dict = {user_id: user_id for user_id in user_ids}
            name_dict.update({user.user_id: str(user.name) for user in users if user.name})
            return name_dict

    except Exception as e:
        logging.exception(f"Failed to fetch users from database: {e}")
        return {user_id: user_id for user_id in user_ids}


async def fetch_user_name(user_id: str) -> str:
    """Fetch a single user name from the database."""
    try:
        engine = get_async_engine()
        async with AsyncSession(engine) as session:
            query = select(User).where(
                User.user_id == user_id,  # type: ignore
            )
            result = await session.execute(query)
            user = result.scalar_one_or_none()

            return str(user.name) if user else UNKNOWN_USER

    except Exception as e:
        logging.exception(f"Failed to fetch user name from database: {e}")
        return UNKNOWN_USER
