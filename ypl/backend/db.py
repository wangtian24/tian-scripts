import uuid
from collections.abc import Generator
from contextvars import ContextVar
from typing import Annotated

from fastapi import Depends
from sqlalchemy import ClauseElement, Compiled, Engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import Session, create_engine

from ypl.backend.config import settings

engine: Engine | None = None
async_engine_ctx: ContextVar[AsyncEngine | None] = ContextVar("async_engine", default=None)


def get_engine() -> Engine:
    global engine
    if engine is None:
        engine = create_engine(str(settings.db_url), connect_args={"sslmode": settings.db_ssl_mode})
    return engine


def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


def get_raw_sql(query: ClauseElement) -> Compiled:
    return query.compile(engine, compile_kwargs={"literal_binds": True})


SessionDep = Annotated[Session, Depends(get_db)]


def get_async_engine() -> AsyncEngine:
    if (engine := async_engine_ctx.get()) is None:
        engine = create_async_engine(
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
        async_engine_ctx.set(engine)
    return engine
