import logging

from sqlalchemy import update
from sqlalchemy.exc import DatabaseError, OperationalError
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ypl.backend.db import get_async_session
from ypl.db.users import User


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def unsubscribe_from_marketing_emails(user_id: str) -> None:
    update_statement = update(User).where(User.user_id == user_id).values(unsubscribed_from_marketing=True)  # type: ignore

    async with get_async_session() as session:
        result = await session.execute(update_statement)
        if result.rowcount == 0:  # type: ignore
            raise ValueError(f"User not found for user {user_id}")
        await session.commit()
