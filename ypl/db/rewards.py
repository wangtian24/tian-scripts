import uuid
from enum import Enum
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel

if TYPE_CHECKING:
    from ypl.db.chats import Eval, Turn
    from ypl.db.point_transactions import PointTransaction
    from ypl.db.users import User


class RewardActionEnum(Enum):
    SIGN_UP = "sign_up"
    PROMPT = "prompt"
    EVALUATION = "evaluation"


class RewardActionLog(BaseModel, table=True):
    __tablename__ = "reward_action_logs"

    reward_action_log_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)

    user_id: str = Field(foreign_key="users.user_id", nullable=False, index=True)
    user: "User" = Relationship(back_populates="reward_action_logs")

    action_type: RewardActionEnum = Field(nullable=False)

    # The turn ID that is associated with this action.
    # Only set if the action type is "evaluation" or "prompt".
    turn_id: uuid.UUID | None = Field(foreign_key="turns.turn_id", default=None, nullable=True)
    turn: "Turn" = Relationship(back_populates="reward_action_logs")

    # The eval ID that is associated with this action.
    # Only set if the action type is "evaluation".
    eval_id: uuid.UUID | None = Field(foreign_key="evals.eval_id", default=None, nullable=True)
    eval: "Eval" = Relationship(back_populates="reward_action_logs")

    # DEPRECATED: Use individual fields like turn_id or eval_id instead.
    # Action type to identifier mapping:
    # - "sign_up": "referrer_id"
    # - "prompt": "prompt_id"
    # - "evaluation": "eval_id" and "turn_id"
    action_details: dict[str, str] = Field(default_factory=dict, sa_type=sa.JSON)

    # If we reward a user for an action, we store the reward id here.
    # Multiple actions can be rewarded with the same reward id.
    associated_reward_id: uuid.UUID | None = Field(foreign_key="rewards.reward_id", default=None, nullable=True)
    associated_reward: "Reward" = Relationship(back_populates="reward_action_logs")


class RewardStatusEnum(Enum):
    UNCLAIMED = "unclaimed"
    CLAIMED = "claimed"
    REJECTED = "rejected"


class Reward(BaseModel, table=True):
    __tablename__ = "rewards"

    reward_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)
    user_id: str = Field(foreign_key="users.user_id", nullable=False, index=True)
    user: "User" = Relationship(back_populates="rewards")

    # The number of credits to be rewarded.
    credit_delta: int = Field(nullable=False, default=0)

    # The status of the reward.
    # Defaults to unclaimed.
    status: RewardStatusEnum = Field(
        sa_column=sa.Column(
            sa.Enum(RewardStatusEnum),
            default=RewardStatusEnum.UNCLAIMED,
            server_default=RewardStatusEnum.UNCLAIMED.name,
            nullable=False,
        )
    )

    # The reason for this reward.
    # Can be changed to an enum if we want to structure this better.
    # Leaving it as a string for now to keep it flexible while the incentive model is being figured out.
    reason: str = Field(nullable=False)

    # The reward action log ids that are associated with this reward.
    reward_action_logs: list["RewardActionLog"] = Relationship(back_populates="associated_reward")

    # Transaction associated with claiming the reward.
    claim_transaction: "PointTransaction" = Relationship(back_populates="claimed_reward")

    # The turn ID for which the reward is given.
    # Only set if the reward is for an evaluation.
    turn_id: uuid.UUID | None = Field(foreign_key="turns.turn_id", default=None, nullable=True)
    turn: "Turn" = Relationship(back_populates="rewards")
