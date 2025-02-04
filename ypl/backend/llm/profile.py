import logging

from pydantic import BaseModel
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlmodel import select
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ypl.backend.db import get_async_session
from ypl.db.users import User


class UserProfileResponse(BaseModel):
    """Represents the response of a user profile."""

    user_id: str
    name: str | None
    country_code: str | None
    discord_username: str | None
    educational_institution: str | None
    city: str | None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def get_user_profile(user_id: str) -> UserProfileResponse | None:
    query = select(User).where(
        User.user_id == user_id,
        User.deleted_at.is_(None),  # type: ignore
    )

    async with get_async_session() as session:
        result = await session.exec(query)
        user = result.one_or_none()
        if user is None:
            return None
        return UserProfileResponse(
            user_id=user.user_id,
            name=user.name,
            country_code=user.country_code,
            discord_username=user.discord_username,
            educational_institution=user.educational_institution,
            city=user.city,
        )
