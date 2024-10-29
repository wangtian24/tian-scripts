import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from business_rules import run_all
from business_rules.variables import BaseVariables, boolean_rule_variable, numeric_rule_variable
from sqlmodel import Field, Relationship

from ypl.db.base import BaseModel

if TYPE_CHECKING:
    from ypl.db.chats import Eval, Turn
    from ypl.db.point_transactions import PointTransaction
    from ypl.db.users import User


class RewardActionEnum(Enum):
    SIGN_UP = "sign_up"
    # Deprecated. Use TURN instead.
    PROMPT = "prompt"
    EVALUATION = "evaluation"
    TURN = "turn"


class RewardActionLog(BaseModel, table=True):
    __tablename__ = "reward_action_logs"

    reward_action_log_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)

    user_id: str = Field(foreign_key="users.user_id", nullable=False, index=True)
    user: "User" = Relationship(back_populates="reward_action_logs")

    action_type: RewardActionEnum = Field(nullable=False)

    # The turn ID that is associated with this action.
    # Only set if the action type is "evaluation" or "turn".
    turn_id: uuid.UUID | None = Field(foreign_key="turns.turn_id", default=None, nullable=True)
    turn: "Turn" = Relationship(back_populates="reward_action_logs")

    # The eval ID that is associated with this action.
    # Only set if the action type is "evaluation".
    eval_id: uuid.UUID | None = Field(foreign_key="evals.eval_id", default=None, nullable=True)
    eval: "Eval" = Relationship(back_populates="reward_action_logs")

    # DEPRECATED: Use individual fields like turn_id or eval_id instead.
    # Action type to identifier mapping:
    # - "sign_up": "referrer_id"
    # - "turn": "turn_id"
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

    # The rules that triggered this reward.
    reward_probability_rule_id: uuid.UUID | None = Field(
        foreign_key="reward_probability_rules.reward_probability_rule_id", default=None, nullable=True
    )
    reward_probability_rule: "RewardProbabilityRule" = Relationship(back_populates="rewards")
    reward_amount_rule_id: uuid.UUID | None = Field(
        foreign_key="reward_amount_rules.reward_amount_rule_id", default=None, nullable=True
    )
    reward_amount_rule: "RewardAmountRule" = Relationship(back_populates="rewards")


class RewardVariables(BaseVariables):  # type: ignore
    def __init__(self, context: dict[str, Any]):
        self.context = context

    @boolean_rule_variable
    def is_new_user(self) -> bool | None:
        return self.context.get("is_new_user")

    @boolean_rule_variable
    def is_inactive_user(self) -> bool | None:
        return self.context.get("is_inactive_user")

    @boolean_rule_variable
    def is_first_turn(self) -> bool | None:
        return self.context.get("is_first_turn")

    @numeric_rule_variable
    def credits(self) -> float | None:
        return self.context.get("points")

    @numeric_rule_variable
    def turn_quality_score(self) -> float | None:
        return self.context.get("turn_quality_score")


class RewardRule(BaseModel, table=False):
    __abstract__ = True

    name: str = Field(nullable=False)

    # The priority of the rule. Rules with higher values are applied first.
    priority: int = Field(nullable=False)

    # Whether the rule should be applied.
    is_active: bool = Field(nullable=False, default=True)

    # Whether the rule is the default rule. Only one rule should be marked as default.
    is_default: bool = Field(nullable=False, default=False)

    # The matching conditions, in the format specified by https://github.com/venmo/business-rules.
    conditions: dict = Field(sa_type=sa.JSON, default_factory=dict)

    def matches(self, context: dict[str, Any]) -> bool:
        """Returns True if the rule matches the context."""
        if not self.is_active:
            return False

        return bool(
            run_all(
                rule_list=[{"conditions": self.conditions, "actions": []}],
                defined_variables=RewardVariables(context),
                defined_actions=[],
                stop_on_first_trigger=True,
            )
        )


class RewardProbabilityRule(RewardRule, table=True):
    __tablename__ = "reward_probability_rules"

    reward_probability_rule_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)

    # The probability of returning a reward, if the rule matches.
    probability: float = Field(nullable=False)

    rewards: list[Reward] = Relationship(back_populates="reward_probability_rule")


class RewardAmountRule(RewardRule, table=True):
    __tablename__ = "reward_amount_rules"

    reward_amount_rule_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)

    # The range of possible reward amounts, if the rule matches.
    min_value: int = Field(nullable=True)
    max_value: int = Field(nullable=True)

    # The mean reward amount, if the rule matches.
    mean_value: float = Field(nullable=True)

    # Possible comments.
    comments: list[str] = Field(sa_column=sa.Column(sa.ARRAY(sa.String)))

    rewards: list[Reward] = Relationship(back_populates="reward_amount_rule")
