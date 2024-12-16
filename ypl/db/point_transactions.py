import uuid
from enum import Enum
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel

if TYPE_CHECKING:
    from ypl.db.payments import PaymentTransaction
    from ypl.db.rewards import Reward
    from ypl.db.users import User


class PointsActionEnum(Enum):
    UNKNOWN = "unknown"
    SIGN_UP = "sign_up"
    REWARD = "reward"
    CASHOUT = "cashout"
    ADJUSTMENT = "adjustment"

    # Not used anymore. Kept for historical reasons.
    PROMPT = "prompt"
    EVALUATION = "evaluation"


class PointTransaction(BaseModel, table=True):
    __tablename__ = "point_transactions"

    transaction_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    user_id: uuid.UUID = Field(foreign_key="users.user_id", nullable=False, sa_type=sa.Text)
    user: "User" = Relationship(back_populates="point_transactions")

    # Can be negative or postiive depending on the action.
    point_delta: int = Field(nullable=False, default=0)

    action_type: PointsActionEnum = Field(nullable=False)
    # Action type to identifier mapping:
    # - "sign_up": "referrer_id"
    # - "reward": "reward_id"
    # - "adjustment": "adjustment_reason"
    action_details: dict[str, str] = Field(default_factory=dict, sa_type=sa.JSON)

    # Set if this transaction is associated with a reward.
    claimed_reward_id: uuid.UUID | None = Field(foreign_key="rewards.reward_id", nullable=True)
    claimed_reward: "Reward" = Relationship(back_populates="claim_transaction")

    # Set if this transaction is associated with a cashout.
    cashout_payment_transaction_id: uuid.UUID | None = Field(
        foreign_key="payment_transactions.payment_transaction_id", nullable=True
    )
    cashout_payment_transaction: "PaymentTransaction" = Relationship(back_populates="credits_transaction")

    # Needed for Column(JSON)
    class Config:
        arbitrary_types_allowed = True
