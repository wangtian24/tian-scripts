import logging
import math
import random
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Literal
from uuid import UUID

from cachetools.func import ttl_cache
from pydantic import BaseModel
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlmodel import Session, select, update
from sqlmodel.ext.asyncio.session import AsyncSession
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ypl.backend.db import get_async_engine, get_engine
from ypl.db.chats import Turn, TurnQuality
from ypl.db.point_transactions import PointsActionEnum, PointTransaction
from ypl.db.rewards import (
    Reward,
    RewardActionLog,
    RewardAmountRule,
    RewardProbabilityRule,
    RewardRule,
    RewardStatusEnum,
)
from ypl.db.users import User

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The base probability of receiving a reward for a turn
BASE_REWARD_PROBABILITY = 0.8

# Define mean reward for evals, baseline value for medium tier (method="mean")
MEAN_EVAL_REWARD = 50


@dataclass
class RewardTier:
    # The minimum quality score (scale of 10) required for this tier
    quality_threshold: int
    # The range of possible reward values (min, max)
    reward_range: tuple[int, int]
    # Mean reward for this tier
    mean_reward: float
    # List of possible comments to be given for rewards in this tier
    comments: list[str]
    name: str


REWARD_TIER_VERY_LOW = "very_low"
REWARD_TIER_LOW = "low"
REWARD_TIER_MEDIUM = "medium"
REWARD_TIER_HIGH = "high"
# Users with higher point balances will receive rewards at a lower rate/amount.
MAX_POINTS = 20000

DEFAULT_COMMENTS = [
    "Keep it up!",
    "Keep going!",
    "Keep engaging for more rewards!",
]


@dataclass
class RewardCreationResponse:
    is_rewarded: bool = False
    # Reward ID for the client to use while claiming the reward.
    reward_id: UUID | None = None
    comment: str | None = None
    credit_delta: int | None = None


def get_matching_rule(rules: list[RewardRule], context: dict[str, Any]) -> RewardRule | None:
    """Get the first matching reward rule in `rules`, given the variables in `context`.

    If no rule matches, return the default rule.
    If no default rule exists, return None.
    """

    default_rule = None
    for rule in rules:
        if rule.is_default:
            default_rule = rule
        if rule.conditions and rule.matches(context):
            return rule
    return default_rule


@dataclass
class UserTurnReward:
    user_id: str
    turn_id: UUID
    is_first_turn: bool = False
    is_new_user: bool = False
    is_inactive_user: bool = False
    turn_quality_score: float = -1
    points: int = 0
    amount_rule: RewardAmountRule | None = None
    probability_rule: RewardProbabilityRule | None = None

    def __post_init__(self) -> None:
        self._fetch_data_and_set_flags()

    def _fetch_data_and_set_flags(self) -> None:
        with Session(get_engine()) as session:
            result = session.exec(
                select(User, TurnQuality)
                .join(Turn, Turn.creator_user_id == User.user_id)  # type: ignore
                .where(User.user_id == self.user_id, Turn.turn_id == self.turn_id)
                .join(TurnQuality, TurnQuality.turn_id == Turn.turn_id, isouter=True)  # type: ignore
            ).first()

            if not result:
                logger.warning(f"No data found for turn_id: {self.turn_id}")
                return

            user, turn_quality = result

            self.is_new_user = user.is_new_user()
            self.is_inactive_user = user.is_inactive_user()
            self.points = user.points

            first_turn_id = session.exec(
                select(Turn.turn_id).where(Turn.creator_user_id == self.user_id).order_by(Turn.created_at)  # type: ignore
            ).first()
            self.is_first_turn = first_turn_id == self.turn_id

            overall_quality = turn_quality.get_overall_quality()
            if overall_quality is not None:
                self.turn_quality_score = overall_quality

            self.amount_rule = self._get_amount_rule()
            self.probability_rule = self._get_probability_rule()

    def _get_amount_rule(self) -> RewardAmountRule | None:
        return get_matching_rule(get_reward_amount_rules(), asdict(self))  # type: ignore

    def _get_probability_rule(self) -> RewardProbabilityRule | None:
        return get_matching_rule(get_reward_probability_rules(), asdict(self))  # type: ignore

    def get_amount(self, method: Literal["range", "mean"] = "range") -> int:
        rule = self.amount_rule
        if not rule:
            logger.warning(f"No reward amount rule found for turn_id: {self.turn_id}")
            return 0
        return (
            get_reward(min_value=rule.min_value, max_value=rule.max_value)
            if method == "range"
            else get_reward(rule.mean_value)
        )

    def get_probability(self) -> float:
        rule = self.probability_rule
        if not rule:
            logger.warning(f"No reward probability rule found for turn_id: {self.turn_id}")
            return 0
        return rule.probability

    def get_reward_comment(self) -> str:
        if self.turn_quality_score is None:
            return random.choice(DEFAULT_COMMENTS)

        if self.amount_rule is None:
            return "Keep engaging for more rewards!"

        return random.choice(self.amount_rule.comments)


def get_reward(
    mean_reward: float = MEAN_EVAL_REWARD, min_value: int | None = None, max_value: int | None = None
) -> int:
    raw_reward = math.ceil(-mean_reward * math.log(1 - random.random()))

    if min_value is not None and max_value is not None:
        raw_reward_range = math.ceil(-mean_reward * math.log(1e-5))
        raw_reward = max(0, min(raw_reward, raw_reward_range))
        scaled_reward = min_value + (raw_reward / raw_reward_range) * (max_value - min_value)
        return int(round(scaled_reward, -1))

    return int(round(raw_reward, -1))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
def reward(user_id: str, turn_id: UUID) -> tuple[bool, int, str, RewardAmountRule | None, RewardProbabilityRule | None]:
    """
    Determine if a user should be rewarded for a turn and calculate the reward amount.

    Args:
        user_id (str): The ID of the user.
        turn_id (UUID): The ID of the turn.

    Returns:
        tuple[bool, int, str]: A tuple containing:
            - bool: Whether the user should be rewarded (True) or not (False).
            - int: The reward amount (in points). 0 if not rewarded.
            - str: A comment or message about the reward.
    """
    user_turn_reward = UserTurnReward(user_id, turn_id)
    reward_probability = user_turn_reward.get_probability()
    should_reward = random.random() < reward_probability

    reward_amount = user_turn_reward.get_amount()
    reward_comment = user_turn_reward.get_reward_comment()

    return (
        should_reward,
        reward_amount,
        reward_comment,
        user_turn_reward.amount_rule,
        user_turn_reward.probability_rule,
    )


@dataclass
class RewardClaimedResponse:
    # TODO(arawind): Stop using reason.
    reason: str
    comment: str
    credit_delta: int
    current_credit_balance: int
    status: RewardStatusEnum


class RewardClaimStruct(BaseModel):
    status: RewardStatusEnum
    comment: str
    credit_delta: int
    current_credit_balance: int


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def create_reward_action_log(reward_action_log: RewardActionLog) -> RewardActionLog:
    async with AsyncSession(get_async_engine()) as session:
        session.add(reward_action_log)
        await session.commit()
        await session.refresh(reward_action_log)
        return reward_action_log


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def create_reward(
    user_id: str,
    credit_delta: int,
    comment: str,
    reward_action_logs: list[RewardActionLog],
    turn_id: UUID | None = None,
    reward_amount_rule: RewardAmountRule | None = None,
    reward_probability_rule: RewardProbabilityRule | None = None,
) -> Reward:
    async with AsyncSession(get_async_engine()) as session:
        amount_rule_id = reward_amount_rule.reward_amount_rule_id if reward_amount_rule else None
        probability_rule_id = reward_probability_rule.reward_probability_rule_id if reward_probability_rule else None
        reward = Reward(
            user_id=user_id,
            credit_delta=credit_delta,
            reason=comment,
            turn_id=turn_id,
            reward_amount_rule_id=amount_rule_id,
            reward_probability_rule_id=probability_rule_id,
        )
        async with session.begin():
            session.add(reward)

            for reward_action_log in reward_action_logs:
                reward_action_log.associated_reward_id = reward.reward_id
                session.add(reward_action_log)

        await session.refresh(reward)
        return reward


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def process_reward_claim(reward_id: UUID, user_id: str) -> RewardClaimStruct:
    """
    Processes the reward claim and returns the current credit balance of the user.

    Atomically:
    1. Create a credit transaction for the reward.
    2. Increment the user's credit balance.
    3. Update the reward status to claimed.
    """
    async with AsyncSession(get_async_engine()) as session:
        async with session.begin():
            # Set the isolation level to SERIALIZABLE to prevent concurrent updates of the credit balance.
            await session.connection(execution_options={"isolation_level": "SERIALIZABLE"})
            reward_query = (
                select(Reward, User)
                .join(User)
                .where(
                    Reward.reward_id == reward_id,
                    Reward.deleted_at.is_(None),  # type: ignore
                    Reward.user_id == user_id,
                    Reward.status != RewardStatusEnum.REJECTED,
                )
            )
            result = await session.exec(reward_query)
            reward, user = result.one()

            # Exit early if the reward status is not unclaimed.
            # To keep it idempotent, we return all the info that we get when the reward is newly claimed.
            if reward.status != RewardStatusEnum.UNCLAIMED:
                return RewardClaimStruct(
                    status=reward.status,
                    comment=reward.reason,
                    credit_delta=reward.credit_delta,
                    current_credit_balance=user.points,
                )

            # 1. Create a credit transaction for the reward.
            credit_transaction = PointTransaction(
                transaction_id=uuid.uuid4(),
                user_id=user_id,
                point_delta=reward.credit_delta,
                action_type=PointsActionEnum.REWARD,
                action_details={"reward_id": str(reward.reward_id)},
                claimed_reward_id=reward.reward_id,
            )
            session.add(credit_transaction)

            # 2. Increment the user's credit balance.
            inc_user_credits_stmt = (
                update(User)
                .returning(User.points)  # type: ignore
                .where(
                    User.user_id == user_id,
                    User.deleted_at.is_(None),  # type: ignore
                )
                .values(points=User.points + reward.credit_delta)
            )
            result = await session.exec(inc_user_credits_stmt)
            row = result.one()
            new_credit_balance = int(row.points)  # type: ignore

            # 3. Update the reward status to claimed.
            reward.status = RewardStatusEnum.CLAIMED
            session.add(reward)

        await session.refresh(reward)

        return RewardClaimStruct(
            status=reward.status,
            comment=reward.reason,
            credit_delta=reward.credit_delta,
            current_credit_balance=new_credit_balance,
        )


def _get_reward_rules(rule_class: type[RewardAmountRule] | type[RewardProbabilityRule]) -> list[RewardRule]:
    """Get active reward rules of the specified type, sorted by priority."""
    with Session(get_engine()) as session:
        rules = session.exec(
            select(rule_class)
            .where(
                rule_class.is_active.is_(True),  # type: ignore
                rule_class.deleted_at.is_(None),  # type: ignore
            )
            .order_by(rule_class.priority.desc())  # type: ignore
        ).all()

        num_default_rules = len([r for r in rules if r.is_default])
        if num_default_rules == 0:
            logger.error(f"No default {rule_class.__name__} found.")
        elif num_default_rules > 1:
            logger.error(f"Multiple default {rule_class.__name__} found.")

        return list(rules)


@ttl_cache(ttl=600)  # 10 minute cache
def get_reward_amount_rules() -> list[RewardRule]:
    return _get_reward_rules(RewardAmountRule)


@ttl_cache(ttl=600)  # 10 minute cache
def get_reward_probability_rules() -> list[RewardRule]:
    return _get_reward_rules(RewardProbabilityRule)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
