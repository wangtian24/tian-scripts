import logging
import math
import random
import uuid
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlmodel import Session, select, update
from sqlmodel.ext.asyncio.session import AsyncSession
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ypl.backend.db import get_async_engine, get_engine
from ypl.db.chats import Turn, TurnQuality
from ypl.db.point_transactions import PointsActionEnum, PointTransaction
from ypl.db.rewards import Reward, RewardActionLog, RewardStatusEnum
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

# Reward tiers: 1-3, 4-7, 8-10
REWARD_TIERS = {
    REWARD_TIER_HIGH: RewardTier(
        name=REWARD_TIER_HIGH,
        quality_threshold=8,
        reward_range=(200, 1000),
        mean_reward=MEAN_EVAL_REWARD * 1.5,
        comments=[
            "This engages the model in a well-rounded, thought-provoking way.",
            "This prompt is a great conversation starter.",
            "This prompt challenges the model effectively across many aspects.",
        ],
    ),
    REWARD_TIER_MEDIUM: RewardTier(
        name=REWARD_TIER_MEDIUM,
        quality_threshold=4,
        reward_range=(50, 200),
        mean_reward=MEAN_EVAL_REWARD * 1.0,
        comments=[
            "Try writing a more novel prompt for better rewards.",
            "Try adding more complexity to future prompts to earn more rewards.",
            "Try more differentiated prompts for higher reward.",
        ],
    ),
    REWARD_TIER_LOW: RewardTier(
        name=REWARD_TIER_LOW,
        quality_threshold=1,
        reward_range=(10, 50),
        mean_reward=MEAN_EVAL_REWARD * 0.5,
        comments=[
            "Thank you for participating.",
        ],
    ),
    REWARD_TIER_VERY_LOW: RewardTier(
        name=REWARD_TIER_VERY_LOW,
        quality_threshold=10,
        reward_range=(0, 10),
        mean_reward=0,
        comments=[
            "Thank you for participating.",
        ],
    ),
}

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


@dataclass
class UserTurnReward:
    user_id: str
    turn_id: UUID
    is_first_turn: bool = False
    is_new_user: bool = False
    is_inactive_user: bool = False
    turn_quality_score: float | None = None
    points: int = 0

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
                logger.warning(f"No data found for user_id: {self.user_id} and turn_id: {self.turn_id}")
                return

            user, turn_quality = result

            self.is_new_user = user.is_new_user()
            self.is_inactive_user = user.is_inactive_user()
            self.points = user.points

            first_turn_id = session.exec(
                select(Turn.turn_id).where(Turn.creator_user_id == self.user_id).order_by(Turn.created_at)  # type: ignore
            ).first()
            self.is_first_turn = first_turn_id == self.turn_id

            self.turn_quality_score = turn_quality.get_overall_quality() if turn_quality else None

    def calculate_reward_probability(self) -> float:
        if self.points > MAX_POINTS:
            return 0.05
        if self.is_first_turn:
            return 1.0
        base_probability = 0.9 if self.is_new_user or self.is_inactive_user else BASE_REWARD_PROBABILITY
        return min(base_probability * random.uniform(0.8, 1.2), 1.0)

    def get_tier(self) -> RewardTier:
        if self.points > MAX_POINTS:
            return REWARD_TIERS[REWARD_TIER_VERY_LOW]
        if self.turn_quality_score is None:
            return REWARD_TIERS[REWARD_TIER_MEDIUM]
        for tier in REWARD_TIERS.values():
            if self.turn_quality_score >= tier.quality_threshold:
                return tier
        return REWARD_TIERS[REWARD_TIER_LOW]

    def get_tiered_reward(self, method: Literal["range", "mean"] = "range") -> int:
        """
        Get tiered reward amount based on turn quality score, using either range or mean of reward.
        """
        tier = self.get_tier()
        min_value, max_value = tier.reward_range
        return (
            get_reward(min_value=min_value, max_value=max_value) if method == "range" else get_reward(tier.mean_reward)
        )

    def get_reward_comment(self) -> str:
        if self.turn_quality_score is None:
            return random.choice(DEFAULT_COMMENTS)

        return random.choice(self.get_tier().comments)


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
def reward(user_id: str, turn_id: UUID) -> tuple[bool, int, str]:
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
    reward_probability = user_turn_reward.calculate_reward_probability()

    should_reward = random.random() < reward_probability
    reward_amount = user_turn_reward.get_tiered_reward() if should_reward else 0
    reward_comment = user_turn_reward.get_reward_comment() if should_reward else "Keep engaging for more rewards!"

    return should_reward, reward_amount, reward_comment


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
) -> Reward:
    async with AsyncSession(get_async_engine()) as session:
        reward = Reward(user_id=user_id, credit_delta=credit_delta, reason=comment, turn_id=turn_id)
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
