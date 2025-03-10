import uuid

from sqlmodel import Field

from ypl.db.base import BaseModel


class RoutingReason(BaseModel, table=True):
    __tablename__ = "routing_reasons"

    routing_reason_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    reason: str | None = Field(nullable=False, primary_key=True, index=True)
    description: str | None = Field(nullable=False)
    is_active: bool | None = Field(nullable=True, default=True, index=True)
