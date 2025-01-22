import logging

from sqlalchemy.exc import DatabaseError, OperationalError
from sqlmodel import select
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ypl.backend.db import get_async_session
from ypl.db.users import UserProfile


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def get_user_profile(user_id: str) -> UserProfile | None:
    query = select(UserProfile).where(
        UserProfile.user_id == user_id,
        UserProfile.deleted_at.is_(None),  # type: ignore
    )

    async with get_async_session() as session:
        result = await session.exec(query)
        return result.one_or_none()
