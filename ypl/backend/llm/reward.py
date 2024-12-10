import asyncio
import logging
import math
import random
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Literal
from uuid import UUID

from cachetools.func import ttl_cache
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from sqlalchemy import and_, func
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlmodel import Session, select, update
from sqlmodel.ext.asyncio.session import AsyncSession
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ypl.backend.config import settings
from ypl.backend.db import get_async_engine, get_engine
from ypl.backend.llm.chat import ModelInfo, get_chat_model
from ypl.backend.llm.constants import ChatProvider
from ypl.backend.llm.judge import FeedbackQualityLabeler
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import Chat, Eval, Turn, TurnQuality
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

load_dotenv()  # Load environment variables from .env file

# Define mean reward for evals, baseline value for medium tier (method="mean")
MEAN_EVAL_REWARD = 50
FEEDBACK_REWARD_LOWER_BOUND = 300
FEEDBACK_REWARD_UPPER_BOUND = 2000
QT_EVAL_REWARD_LOWER_BOUND = 0
QT_EVAL_REWARD_UPPER_BOUND = 300

VERY_POOR_FEEDBACK_SCORE = 1
POOR_FEEDBACK_SCORE = 2
AVERAGE_FEEDBACK_SCORE = 3
GOOD_FEEDBACK_SCORE = 4
EXCELLENT_FEEDBACK_SCORE = 5

FEEDBACK_QUALITY_MULTIPLIER = {
    # Poor quality (1)
    1: 0.25,  # ~300-400 range
    # Below average quality (2)
    2: 0.35,  # ~400-600 range
    # Average quality (3)
    3: 0.45,  # ~600-800 range
    # Good quality (4)
    4: 0.75,  # ~1000-1500 range
    # Excellent quality (5)
    5: 1.0,  # ~1500-2000 range
}


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
    "Thanks for your input on model responses.",
    "Keep engaging for more rewards.",
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
    # TODO(carmen): Deprecate turn_position_in_chat and is_first_turn.
    turn_position_in_chat: int = 0
    is_first_turn: bool = False
    previous_chat_count: int = 0
    previous_eval_count_in_chat: int = 0
    is_first_eval: bool = False
    is_new_user: bool = False
    is_inactive_user: bool = False
    turn_quality_score: float = -1
    points: int = 0
    amount_rule: RewardAmountRule | None = None
    probability_rule: RewardProbabilityRule | None = None
    points_last_day: int = 0
    points_last_week: int = 0
    points_last_month: int = 0

    def __post_init__(self) -> None:
        self._fetch_data_and_set_flags()

    def _fetch_data_and_set_flags(self) -> None:
        with Session(get_engine()) as session:
            result = session.exec(
                select(User, Chat.chat_id, Chat.created_at, Turn.created_at, Turn.sequence_id, TurnQuality)  # type: ignore
                .join(Chat, Chat.creator_user_id == User.user_id)
                .join(Turn)
                .join(TurnQuality, isouter=True)
                .where(Turn.turn_id == self.turn_id)
                .order_by(Chat.created_at)
            ).first()

            if not result:
                logging.warning(
                    json_dumps(
                        {
                            "message": "No data found for turn_id",
                            "turn_id": str(self.turn_id),
                        }
                    )
                )
                return

            # TODO(carmen): Deprecate turn_position_in_chat
            user, chat_id, chat_created_at, turn_created_at, self.turn_position_in_chat, turn_quality = result

            self.is_new_user = user.is_new_user()
            self.is_inactive_user = user.is_inactive_user()
            self.points = user.points

            self.is_first_eval = not session.exec(
                select(
                    func.exists(
                        select(1)
                        .select_from(Eval)
                        .join(Turn)
                        .where(
                            Eval.user_id == self.user_id,
                            Turn.created_at < turn_created_at,
                        )
                    )
                )
            ).one()

            self.previous_eval_count_in_chat = session.exec(
                select(func.count())
                .select_from(Eval)
                .join(Turn)
                .where(
                    Turn.chat_id == chat_id,
                    Eval.user_id == self.user_id,
                    Turn.created_at < turn_created_at,
                )
            ).one()

            self.previous_chat_count = session.exec(
                select(func.count())
                .select_from(Chat)
                .where(
                    Chat.creator_user_id == self.user_id,
                    Chat.created_at < (chat_created_at if chat_created_at else None),
                )
            ).one()

            # TODO(carmen): Deprecate is_first_turn.
            first_turn_id = session.exec(
                select(Turn.turn_id).where(Turn.creator_user_id == self.user_id).order_by(Turn.created_at)  # type: ignore
            ).first()
            self.is_first_turn = first_turn_id == self.turn_id

            # Set turn quality if available
            if turn_quality:
                self.turn_quality_score = turn_quality.get_overall_quality() or -1

            self.amount_rule = self._get_amount_rule()
            self.probability_rule = self._get_probability_rule()

            self.points_last_day = self._get_reward_points(session, timedelta(days=1))
            self.points_last_week = self._get_reward_points(session, timedelta(days=7))
            self.points_last_month = self._get_reward_points(session, timedelta(days=30))

    def _get_reward_points(self, session: Session, delta: timedelta) -> int:
        result: int | None = session.exec(
            select(func.sum(PointTransaction.point_delta)).where(
                PointTransaction.user_id == self.user_id,
                PointTransaction.deleted_at.is_(None),  # type: ignore
                PointTransaction.action_type == PointsActionEnum.REWARD,
                PointTransaction.created_at > (datetime.now() - delta),  # type: ignore
            )
        ).one()
        return result or 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with UUID fields converted to strings."""
        d = asdict(self)
        d["turn_id"] = str(self.turn_id)
        return d

    def _get_amount_rule(self) -> RewardAmountRule | None:
        return get_matching_rule(get_reward_amount_rules(), self.to_dict())  # type: ignore

    def _get_probability_rule(self) -> RewardProbabilityRule | None:
        return get_matching_rule(get_reward_probability_rules(), self.to_dict())  # type: ignore

    def get_amount(self, method: Literal["range", "mean"] = "range") -> int:
        rule = self.amount_rule
        if not rule:
            log_dict = {
                "message": "No reward amount rule found for turn_id",
                "turn_id": self.turn_id,
            }
            logging.warning(json_dumps(log_dict))
            return 0
        return (
            get_reward(min_value=rule.min_value, max_value=rule.max_value)
            if method == "range"
            else get_reward(rule.mean_value)
        )

    def get_probability(self) -> float:
        rule = self.probability_rule
        if not rule:
            log_dict = {
                "message": "No reward probability rule found for turn_id",
                "turn_id": str(self.turn_id),
            }
            logging.warning(json_dumps(log_dict))
            return 0
        return rule.probability

    def get_reward_comment(self) -> str:
        if self.amount_rule is None:
            return random.choice(DEFAULT_COMMENTS)

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
def turn_based_reward(
    user_id: str, turn_id: UUID
) -> tuple[bool, int, str, RewardAmountRule | None, RewardProbabilityRule | None]:
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

    # Override reward amount for local environments
    if settings.ENVIRONMENT == "local":
        reward_amount = random.randint(1, 10)
        should_reward = True

    # A safety check to prevent negative or zero credit rewards from being given.
    if reward_amount <= 0:
        should_reward = False
        reward_amount = 0

    return (
        should_reward,
        reward_amount,
        reward_comment,
        user_turn_reward.amount_rule,
        user_turn_reward.probability_rule,
    )


def get_reward_llm() -> BaseChatModel:
    """Get the LLM for reward evaluation. Separated for easier testing."""
    return get_chat_model(
        ModelInfo(
            provider=ChatProvider.OPENAI,
            model="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
        ),
        temperature=0.0,
    )


# At module level
_cached_llm: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    """Get or create the LLM instance."""
    global _cached_llm
    if _cached_llm is None:
        _cached_llm = get_reward_llm()
    return _cached_llm


# Update get_feedback_quality_score to use the function
async def get_feedback_quality_score(user_id: str, feedback: str, llm: BaseChatModel | None = None) -> int:
    try:
        start_time = time.time()
        labeler = FeedbackQualityLabeler(llm or get_llm())

        # Wrap the label call with timeout
        try:
            score = await asyncio.wait_for(labeler.alabel(feedback), timeout=0.5)  # 500ms timeout
        except TimeoutError:
            log_dict = {
                "message": "Timeout getting feedback quality score",
                "user_id": user_id,
                "feedback": feedback,
                "timeout": 0.5,
            }
            logging.warning(json_dumps(log_dict))
            return 2  # Return lower score on timeout since it might be spam

        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        log_dict = {
            "message": "Feedback quality score latency",
            "latency_ms": elapsed_time,
            "score": score,
            "user_id": user_id,
            "feedback": feedback,
        }
        logging.info(json_dumps(log_dict))

        return score
    except Exception as e:
        log_dict = {
            "message": "Error getting feedback quality score",
            "error": str(e),
            "user_id": user_id,
            "feedback": feedback,
        }
        logging.warning(json_dumps(log_dict))
        return 5  # Return average score on error


async def generate_bounded_reward(lower_bound: int, upper_bound: int, quality_score: int | None = None) -> int:
    """
    Generate a normally distributed random reward amount between lower and upper bounds.
    Adjusts the reward based on feedback quality if feedback is provided.

    Args:
        lower_bound (int): Minimum reward amount
        upper_bound (int): Maximum reward amount
        quality_score (int): The feedback quality score
    Returns:
        int: The generated reward amount, rounded to nearest 10
    """
    mean = (lower_bound + upper_bound) / 2

    # Adjust mean based on feedback quality if provided
    if quality_score:
        # Get multiplier based on quality score (default to 1.0 if score is invalid)
        quality_multiplier = FEEDBACK_QUALITY_MULTIPLIER.get(quality_score, 1.0)

        # Apply quality multiplier to mean
        mean = mean * quality_multiplier

    std_dev = (upper_bound - mean) / 3

    reward_amount = int(round(random.gauss(mean, std_dev), -1))
    return max(lower_bound, min(upper_bound, reward_amount))


async def feedback_based_reward(
    user_id: str, feedback_comment: str
) -> tuple[bool, int, str, RewardAmountRule | None, RewardProbabilityRule | None]:
    """
    Determine if a user should be rewarded for feedback and calculate the reward amount.
    The reward amount is based on both feedback length and quality.

    Args:
        user_id: The ID of the user
        feedback_comment: The feedback comment
    """
    should_reward = True

    # Get quality score for the comment
    quality_score = await get_feedback_quality_score(user_id, feedback_comment)

    reward_amount = await generate_bounded_reward(
        lower_bound=FEEDBACK_REWARD_LOWER_BOUND,
        upper_bound=FEEDBACK_REWARD_UPPER_BOUND,
        quality_score=quality_score,
    )

    # Customize comment based on quality
    if quality_score >= EXCELLENT_FEEDBACK_SCORE:
        reward_comment = f"Excellent feedback! Reward: {reward_amount} credits"
    elif quality_score >= GOOD_FEEDBACK_SCORE:
        reward_comment = f"Good quality feedback. Reward: {reward_amount} credits"
    elif quality_score >= AVERAGE_FEEDBACK_SCORE:
        reward_comment = f"Average feedback. Reward: {reward_amount} credits"
    else:
        reward_comment = f"Feedback reward: {reward_amount} credits. More detailed feedback earns higher rewards!"

    log_dict = {
        "user_id": user_id,
        "message": "Feedback based reward",
        "should_reward": should_reward,
        "reward_amount": reward_amount,
        "reward_comment": reward_comment,
    }
    logging.info(json_dumps(log_dict))
    return (
        should_reward,
        reward_amount,
        reward_comment,
        None,  # No specific amount rule for feedback for now
        None,  # No specific probability rule for feedback for now
    )


async def qt_eval_reward(user_id: str) -> tuple[bool, int, str, RewardAmountRule | None, RewardProbabilityRule | None]:
    should_reward = True

    # Generate a random reward amount if none provided
    reward_amount = await generate_bounded_reward(
        lower_bound=QT_EVAL_REWARD_LOWER_BOUND, upper_bound=QT_EVAL_REWARD_UPPER_BOUND
    )

    reward_comment = f"QT Eval reward: {reward_amount} credits."

    return (
        should_reward,
        reward_amount,
        reward_comment,
        None,
        None,
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
            await session.flush()  # Ensure reward_id is available

            # Fetch fresh copies of reward action logs
            for reward_action_log in reward_action_logs:
                # Get a fresh copy from the database
                fresh_log = await session.get(RewardActionLog, reward_action_log.reward_action_log_id)
                if fresh_log:
                    fresh_log.associated_reward_id = reward.reward_id
                    session.add(fresh_log)

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
            log_dict = {
                "message": "No default reward rule found.",
                "rule_class": str(rule_class.__name__),
            }
            logging.error(json_dumps(log_dict))
        elif num_default_rules > 1:
            log_dict = {
                "message": "Multiple default reward rules found.",
                "rule_class": str(rule_class.__name__),
            }
            logging.error(json_dumps(log_dict))

        return list(rules)


@ttl_cache(ttl=600)  # 10 minute cache
def get_reward_amount_rules() -> list[RewardRule]:
    return _get_reward_rules(RewardAmountRule)


@ttl_cache(ttl=600)  # 10 minute cache
def get_reward_probability_rules() -> list[RewardRule]:
    return _get_reward_rules(RewardProbabilityRule)


async def get_reward_action_log_by_user_and_turn(
    user_id: str, turn_id: uuid.UUID, action_type: str
) -> RewardActionLog | None:
    """
    Get reward action log entry for a specific user and turn

    Args:
        user_id: The ID of the user
        turn_id: The ID of the turn
        action_type: The type of the action
    Returns:
        RewardActionLog if found, None otherwise
    """
    async with AsyncSession(get_async_engine()) as session:
        query = select(RewardActionLog).where(
            and_(
                RewardActionLog.user_id == user_id,  # type: ignore
                RewardActionLog.turn_id == turn_id,  # type: ignore
                RewardActionLog.action_type == action_type,  # type: ignore
                RewardActionLog.deleted_at.is_(None),  # type: ignore
            )
        )
        result = await session.exec(query)
        return result.one_or_none()
