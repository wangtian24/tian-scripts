import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field

from ypl.db.base import BaseModel

if TYPE_CHECKING:
    pass


# This table stores scribble contents that can be edited in Yupp Soul.


class Scribbles(BaseModel, table=True):
    __tablename__ = "scribbles"

    scribble_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    label: str = Field(nullable=False, index=True, unique=True)
    content: dict | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
        description="Content of the scribble entry",
    )
