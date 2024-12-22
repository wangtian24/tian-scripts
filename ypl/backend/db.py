import uuid
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends
from sqlalchemy import ClauseElement, Compiled, Engine
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel import Session, create_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.config import settings

engine: Engine | None = None
async_engine: AsyncEngine | None = None

direct_engine: Engine | None = None
direct_async_engine: AsyncEngine | None = None


def get_engine() -> Engine:
    global engine
    if engine is None:
        engine = create_engine(
            str(settings.db_url), pool_pre_ping=True, pool_recycle=1800, connect_args={"sslmode": settings.db_ssl_mode}
        )
    return engine


def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


def get_raw_sql(query: ClauseElement) -> Compiled:
    return query.compile(engine, compile_kwargs={"literal_binds": True})


SessionDep = Annotated[Session, Depends(get_db)]


def get_async_engine() -> AsyncEngine:
    global async_engine
    if async_engine is None:
        async_engine = create_async_engine(
            str(settings.db_url_async),
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={
                "ssl": settings.db_ssl_mode,
                # below properties are to make asyncpg work with pgbouncer in transaction mode. Ref : https://github.com/MagicStack/asyncpg/issues/1058
                "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4()}__",
                "statement_cache_size": 0,
                "prepared_statement_cache_size": 0,
            },
        )
    return async_engine


async_session_maker = async_sessionmaker(
    get_async_engine(),
    # Uses the SQLModel AsyncSession class to ensure that the session is compatible with SQLModel
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    session = async_session_maker()
    try:
        yield session
    finally:
        await session.close()
