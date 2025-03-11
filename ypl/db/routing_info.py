import uuid
from typing import Any

from sqlalchemy import ARRAY, Column, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel
from ypl.db.chats import ChatMessage, Turn


class RoutingInfo(BaseModel, table=True):
    """
    This tables records the routing information for a given turn, this includes the input to the routing
    like the model and style selector, the intermediate information generated during the routing
    (like prompt categories), and the outcome of the routing (like the selected models and their reasons).

    This information is linked to the ChatMessages (1:n) and Turns (n:1). Each turn can have multiple routing info
    due to "Show More AIs".
    """

    __tablename__ = "routing_info"

    routing_info_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    # the turn where this routing happened
    turn_id: uuid.UUID = Field(foreign_key="turns.turn_id", nullable=False, index=True)

    # -- the input to the routing
    # he model and style selector used for this turn
    selector: dict[str, Any] = Field(default_factory=dict, sa_type=JSONB, nullable=False)

    # -- the intermediate information generated during the routing
    # the categories we have detected from the prompt
    categories: list[str] | None = Field(sa_column=Column(ARRAY(Text), nullable=True))

    # -- the outcome of the routing
    # the details of the routing, the selected models and their reasons.
    routing_outcome: dict[str, Any] = Field(default_factory=dict, sa_type=JSONB, nullable=False)

    # -- Relationships

    # one RoutingInfo belongs to only one Turn, but multiple RoutingInfo rows might correspond to the same turn
    # due to "Show More AIs"
    turn: Turn = Relationship(back_populates="routing_infos")

    # one RoutingInfo can be linked to multiple ChatMessages as the outcome of the routing.
    messages: list[ChatMessage] = Relationship(back_populates="routing_info")
