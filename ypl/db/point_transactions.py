import uuid
from enum import Enum
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel

if TYPE_CHECKING:
    from ypl.db.rewards import Reward
    from ypl.db.users import User


class PointsActionEnum(Enum):
    UNKNOWN = "unknown"
    SIGN_UP = "sign_up"
    PROMPT = "prompt"
    EVALUATION = "evaluation"
    REWARD = "reward"


class PointTransaction(BaseModel, table=True):
    __tablename__ = "point_transactions"

    transaction_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    user_id: uuid.UUID = Field(foreign_key="users.user_id", nullable=False, sa_type=sa.Text)
    user: "User" = Relationship(back_populates="point_transactions")

    # Can be negative or postiive depending on the action.
    point_delta: int = Field(nullable=False, default=0)

    # Deprecated. Credits are now only added via rewards.
    # TODO(arawind): Drop action_* columns.
    action_type: PointsActionEnum = Field(nullable=False)
    # Action type to identifier mapping:
    # - "sign_up": "referrer_id"
    # - "prompt": "prompt_id"
    # - "evaluation": "eval_id"
    # - "reward": "reward_id"
    action_details: dict[str, str] = Field(default_factory=dict, sa_type=sa.JSON)

    # Set if this transaction is associated with a reward.
    claimed_reward_id: uuid.UUID | None = Field(foreign_key="rewards.reward_id", nullable=True)
    claimed_reward: "Reward" = Relationship(back_populates="claim_transaction")

    # Needed for Column(JSON)
    class Config:
        arbitrary_types_allowed = True
