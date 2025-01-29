from sqlmodel.ext.asyncio.session import AsyncSession
from ypl.backend.db import get_async_engine
from ypl.db.app_feedback import AppFeedback


async def store_app_feedback(app_feedback: AppFeedback) -> None:
    async with AsyncSession(get_async_engine()) as session:
        session.add(app_feedback)
        await session.commit()
