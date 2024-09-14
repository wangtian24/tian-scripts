from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import ClauseElement, Compiled, Engine
from sqlmodel import Session, create_engine

from backend.config import settings

engine: Engine | None = None


def get_engine() -> Engine:
    global engine
    if engine is None:
        engine = create_engine(str(settings.db_url))
    return engine


def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


def get_raw_sql(query: ClauseElement) -> Compiled:
    return query.compile(engine, compile_kwargs={"literal_binds": True})


SessionDep = Annotated[Session, Depends(get_db)]
