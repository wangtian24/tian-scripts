import logging

from sqlalchemy.exc import DatabaseError, OperationalError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ypl.backend.db import get_async_engine
from ypl.db.users import User


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def get_user_credit_balance(user_id: str) -> int:
    query = select(User.points).where(
        User.user_id == user_id,
        User.deleted_at.is_(None),  # type: ignore
    )

    async with AsyncSession(get_async_engine()) as session:
        result = await session.exec(query)
        return result.one()
