from fastapi import APIRouter
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.db import get_async_engine

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    try:
        async with AsyncSession(get_async_engine()) as session:
            result = await session.exec(text("SELECT version_num FROM alembic_version"))  # type: ignore
            version = result.scalar()

            return {"status": "ok", "db_version": version or ""}
    except Exception as e:
        return {"status": "error", "message": str(e)}
