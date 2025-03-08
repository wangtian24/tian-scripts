import asyncio
import logging
import math
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import pytz
import yaml
from cachetools.func import ttl_cache
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy import stats
from sqlalchemy import and_, func
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlmodel import Session, select, update
from sqlmodel.ext.asyncio.session import AsyncSession
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ypl.backend.abuse.content import check_model_feedback_abuse
from ypl.backend.config import settings
from ypl.backend.db import get_async_engine, get_engine
from ypl.backend.feedback.app_feedback import (
    AVERAGE_FEEDBACK_SCORE,
    EXCELLENT_FEEDBACK_SCORE,
    FALLBACK_QUALITY_SCORE,
    GOOD_FEEDBACK_SCORE,
)
from ypl.backend.jobs.tasks import post_to_slack_task
from ypl.backend.llm.turn_quality import LOW_EVAL_QUALITY_SCORE, update_user_eval_quality_scores
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import Chat, ChatMessage, Eval, EvalType, Turn, TurnQuality
from ypl.db.invite_codes import SpecialInviteCode, SpecialInviteCodeClaimLog
from ypl.db.point_transactions import PointsActionEnum, PointTransaction
from ypl.db.rewards import (
    Reward,
    RewardActionEnum,
    RewardActionLog,
    RewardAmountRule,
    RewardProbabilityRule,
    RewardRule,
    RewardStatusEnum,
)
from ypl.db.users import User, WaitlistedUser

# Load environment variables from .env file
load_dotenv()

# Define mean reward for evals, baseline value for medium tier (method="mean")
FEEDBACK_REWARD_LOWER_BOUND = 75
FEEDBACK_REWARD_UPPER_BOUND = 175
QT_EVAL_REWARD_LOWER_BOUND = 50
QT_EVAL_REWARD_UPPER_BOUND = 100

# Number of recent evals to check for low-quality evals.
NUM_RECENT_EVALS_FOR_QUALITY_CHECK = 5
# Number of low-quality evals in the recent evals to consider for zero reward.
NUM_LOW_QUALITY_EVALS_FOR_ZERO_REWARD = 2

# Reward types that count towards point limits.
REWARDABLE_POINT_ACTION_TYPES = [PointsActionEnum.REWARD, PointsActionEnum.ADJUSTMENT]
REFERRAL_REWARD_TYPES = [RewardActionEnum.REFERRAL_BONUS_REFERRED_USER, RewardActionEnum.REFERRAL_BONUS_REFERRER]

# Constants for reward rules; populated from data/reward_rules.yml.
RULE_CONSTANTS: dict[str, Any] = {}
RULES_PATH = "data/reward_rules.yml"

FEEDBACK_QUALITY_MULTIPLIER = {
    # Poor quality (1)
    1: 0.2,
    # Below average quality (2)
    2: 0.35,
    # Average quality (3)
    3: 0.5,
    # Good quality (4)
    4: 0.8,
    # Excellent quality (5)
    5: 1.0,
}


@dataclass
class RewardTier:
    # The minimum quality score (scale of 10) required for this tier
    quality_threshold: int
    # The range of possible reward values (min, max)
    reward_range: tuple[int, int]
    # List of possible comments to be given for rewards in this tier
    comments: list[str]
    name: str


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
    high_value_reward_id: UUID | None = None
    high_value_credit_delta: int | None = None


@dataclass
class RewardStatusUpdateResponse:
    reward_id: UUID
    status: RewardStatusEnum


def get_matching_rule(rules: list[RewardRule], context: dict[str, Any]) -> RewardRule | None:
    """Get the first matching reward rule in `rules`, given the variables in `context`.

    If no rule matches, return the default rule.
    If no default rule exists, return None.
    """

    default_rule = None
    for rule in rules:
        if rule.action_type != context.get("action_type"):
            continue
        if rule.is_default:
            default_rule = rule
        if rule.conditions and rule.matches(context):
            return rule
    return default_rule


@dataclass
class UserTurnReward:
    user_id: str
    turn_id: UUID
    chat_id: UUID | None = None
    # TODO(carmen): Deprecate turn_position_in_chat and is_first_turn.
    turn_position_in_chat: int = -1
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
    points_last_award: int = 0
    referral_points_last_day: int = 0
    referral_points_last_week: int = 0
    referral_points_last_month: int = 0
    action_type: RewardActionEnum = RewardActionEnum.TURN
    user_name: str | None = None
    pref_rewards_count: int = -1
    pref_rewards_count_since_reactivation: int = -1
    decay_factor: float | None = None
    decay_factor_high_value: float | None = None
    recent_eval_quality_scores: list[float] = field(default_factory=list)

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

            user, chat_id, chat_created_at, turn_created_at, self.turn_position_in_chat, turn_quality = result

            self.is_new_user = user.is_new_user()
            self.is_inactive_user = user.is_inactive_user()
            self.points = user.points
            self.user_name = user.name
            self.pref_rewards_count = user.get_pref_rewards_count()
            self.pref_rewards_count_since_reactivation = user.get_pref_rewards_count_since_reactivation(
                end_date=turn_created_at
            )
            self.chat_id = chat_id

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

            # Set turn quality if available
            if turn_quality:
                self.turn_quality_score = turn_quality.get_overall_quality() or -1

            reward_points_summary = _get_reward_points_summary(self.user_id, session)
            self.points_last_day = reward_points_summary["points_last_day"]
            self.points_last_week = reward_points_summary["points_last_week"]
            self.points_last_month = reward_points_summary["points_last_month"]
            self.points_last_award = reward_points_summary["points_last_award"]
            self.referral_points_last_day = reward_points_summary["referral_points_last_day"]
            self.referral_points_last_week = reward_points_summary["referral_points_last_week"]
            self.referral_points_last_month = reward_points_summary["referral_points_last_month"]
            self.recent_eval_quality_scores = _get_recent_eval_quality_scores(self.user_id, session)

            self.amount_rule = self._get_amount_rule()
            self.probability_rule = self._get_probability_rule()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with UUID fields converted to strings."""
        d = asdict(self)
        d["turn_id"] = str(self.turn_id)
        return d

    def should_get_zero_reward(self) -> bool:
        """
        A user gets a zero reward with a given probability, unless:
        - last reward was 0
        - first turn in a chat
        - received less than 3 PREF rewards in total
        - received less than 3 PREF rewards since their last reactivation
        """

        # Check if the user has submitted low quality evals recently.
        low_quality_count = len([score for score in self.recent_eval_quality_scores if score == LOW_EVAL_QUALITY_SCORE])
        if low_quality_count >= NUM_LOW_QUALITY_EVALS_FOR_ZERO_REWARD:
            logging.info(
                json_dumps(
                    {
                        "message": "Zero reward due to multiple recent low quality evals",
                        "user_id": self.user_id,
                        "low_quality_count": str(low_quality_count),
                        "recent_eval_quality_scores": str(self.recent_eval_quality_scores),
                    }
                )
            )
            return True

        if (
            self.points_last_award == 0
            or self.turn_position_in_chat == 0
            or (self.pref_rewards_count >= 0 and self.pref_rewards_count < 3)
            or (self.pref_rewards_count_since_reactivation >= 0 and self.pref_rewards_count_since_reactivation < 3)
        ):
            return False

        zero_reward_probability = RULE_CONSTANTS.get("zero_turn_based_reward_probability", 0.0)
        return random.random() < zero_reward_probability  # type: ignore

    def _get_amount_rule(self) -> RewardAmountRule | None:
        return get_matching_rule(get_reward_amount_rules(self.action_type), self.to_dict())  # type: ignore

    def _get_probability_rule(self) -> RewardProbabilityRule | None:
        return get_matching_rule(get_reward_probability_rules(self.action_type), self.to_dict())  # type: ignore

    def _maybe_decay_amounts(self, min_value: int, max_value: int, high_value: bool) -> tuple[int, int, float]:
        # Default to no decay.
        decay_factor = 1.0
        if "daily_points_limit" in RULE_CONSTANTS:
            daily_points_limit = RULE_CONSTANTS["daily_points_limit"]
            weekly_points_limit = RULE_CONSTANTS["weekly_points_limit"]
            monthly_points_limit = RULE_CONSTANTS["monthly_points_limit"]
            fraction_of_limit = max(
                self.points_last_day / daily_points_limit,
                self.points_last_week / weekly_points_limit,
                self.points_last_month / monthly_points_limit,
                # Referrals are only limited at the monthly level.
                self.referral_points_last_month / monthly_points_limit,
            )
            if fraction_of_limit > 0.5:
                # A higher constant means faster decay.
                decay_factor = math.exp(-6 * (fraction_of_limit - 0.5))
                log_dict = {
                    "message": "Decaying reward amounts",
                    "high_value": high_value,
                    "points_last_day": f"{self.points_last_day}/{daily_points_limit}",
                    "points_last_week": f"{self.points_last_week}/{weekly_points_limit}",
                    "points_last_month": f"{self.points_last_month}/{monthly_points_limit}",
                    "referral_points_last_day": f"{self.referral_points_last_day}/{daily_points_limit}",
                    "referral_points_last_week": f"{self.referral_points_last_week}/{weekly_points_limit}",
                    "referral_points_last_month": f"{self.referral_points_last_month}/{monthly_points_limit}",
                    "fraction_of_limit": str(fraction_of_limit),
                    "original_min_value": str(min_value),
                    "original_max_value": str(max_value),
                    "decay_factor": str(decay_factor),
                    "user_name": self.user_name,
                    "user_id": self.user_id,
                    "turn_id": str(self.turn_id),
                    "chat_id": str(self.chat_id),
                }
                min_value = int(round(min_value * decay_factor, -1))
                max_value = int(round(max_value * decay_factor, -1))
                log_dict["decayed_min_value"] = str(min_value)
                log_dict["decayed_max_value"] = str(max_value)
                logging.info(json_dumps(log_dict))

        return min_value, max_value, decay_factor

    def get_amount(self, high_value: bool = False) -> int:
        rule = self.amount_rule
        if not rule:
            log_dict = {
                "message": "No reward amount rule found for turn_id",
                "turn_id": str(self.turn_id),
                "user_name": self.user_name,
                "user_id": self.user_id,
                "chat_id": str(self.chat_id),
            }
            logging.warning(json_dumps(log_dict))
            return 0

        # Set min, max values using conditional expressions
        min_value = rule.high_min_value if (high_value and rule.high_min_value is not None) else rule.min_value
        max_value = rule.high_max_value if (high_value and rule.high_max_value is not None) else rule.max_value

        # Optionally decay the reward amount, as the daily limit approaches.
        min_value, max_value, decay_factor = self._maybe_decay_amounts(min_value, max_value, high_value)

        if high_value:
            self.decay_factor_high_value = decay_factor
        else:
            self.decay_factor = decay_factor

        # Return the reward based on the specified method
        return get_reward(min_value=min_value, max_value=max_value)

    def get_probability(self) -> float:
        rule = self.probability_rule
        if not rule:
            log_dict = {
                "message": "No reward probability rule found for turn_id",
                "turn_id": str(self.turn_id),
                "user_id": self.user_id,
                "user_name": self.user_name,
                "chat_id": str(self.chat_id),
            }
            logging.warning(json_dumps(log_dict))
            return 0
        return rule.probability

    def get_reward_comment(self) -> str:
        if self.amount_rule is None:
            return random.choice(DEFAULT_COMMENTS)

        return random.choice(self.amount_rule.comments)


def get_reward(min_value: int, max_value: int) -> int:
    # min_value is the mean in a normal distribution
    # max_value is 3 standard deviations up (sigma), and values are clamped thereafter

    if min_value == max_value:
        return min_value
    if max_value < min_value:
        raise ValueError("max_value must be greater than or equal to min_value.")

    sigma = (max_value - min_value) / 3

    a = 0
    b = (max_value - min_value) / sigma

    # Sample from truncated normal (right-side half)
    reward = stats.truncnorm.rvs(a=a, b=b, loc=min_value, scale=sigma, size=1)[0]
    reward = int(math.floor(reward))
    return int(min(reward, max_value))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((TimeoutError,)),
)
async def _update_user_eval_quality_scores(user_id: str) -> None:
    # In most cases, waiting a bit will result in the inclusion of the eval for the current turn.
    # If not, it will be included in the next batch of evals.
    await asyncio.sleep(5)
    await update_user_eval_quality_scores(user_id)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def turn_based_reward(
    user_id: str, turn_id: UUID
) -> tuple[bool, int, int, str, RewardAmountRule | None, RewardProbabilityRule | None]:
    """
    Determine if a user should be rewarded for a turn and calculate both regular and high value reward amounts.

    Args:
        user_id (str): The ID of the user.
        turn_id (UUID): The ID of the turn.

    Returns:
        tuple: A tuple containing:
            - bool: Whether the user should be rewarded (True) or not (False).
            - int: The reward amount (in points). 0 if not rewarded.
            - int: The high value reward amount if the user provides model feedback.
            - str: A comment or message about the reward.
            - RewardAmountRule | None: The reward amount rule used.
            - RewardProbabilityRule | None: The reward probability rule used.
    """

    asyncio.create_task(_update_user_eval_quality_scores(user_id))
    asyncio.create_task(check_model_feedback_abuse(user_id))
    return _handle_turn_based_reward(UserTurnReward(user_id, turn_id))


def _handle_turn_based_reward(
    user_turn_reward: UserTurnReward,
) -> tuple[bool, int, int, str, RewardAmountRule | None, RewardProbabilityRule | None]:
    def _handle_zero_turn_based_reward(user_turn_reward: UserTurnReward) -> tuple[bool, int, int, str, None, None]:
        """Handle the case where zero turn-based reward probability is defined."""
        should_reward = True
        reward_amount = 0
        high_value_reward_amount = user_turn_reward.get_amount(high_value=False)
        if high_value_reward_amount < RULE_CONSTANTS["min_ever_high_value_reward_amount"]:
            high_value_reward_amount = RULE_CONSTANTS["min_ever_high_value_reward_amount"]
        reward_comment = "Better luck next time!"

        log_dict = {
            "message": "Override reward with zero amount",
            "user_id": user_turn_reward.user_id,
            "user_name": user_turn_reward.user_name,
            "reward_amount": reward_amount,
            "high_value_reward_amount": high_value_reward_amount,
            "reward_comment": reward_comment,
            "turn_id": str(user_turn_reward.turn_id),
            "chat_id": str(user_turn_reward.chat_id),
        }
        logging.info(json_dumps(log_dict))

        log_reward_debug_info(
            action_type=RewardActionEnum.TURN,
            user_id=user_turn_reward.user_id,
            user_name=user_turn_reward.user_name,
            should_reward=should_reward,
            reward_amount=reward_amount,
            reward_comment=reward_comment,
            probability_rule=None,
            amount_rule=None,
            high_value_reward_amount=high_value_reward_amount,
            decay_factor=user_turn_reward.decay_factor,
            decay_factor_high_value=user_turn_reward.decay_factor_high_value,
            turn_id=str(user_turn_reward.turn_id),
            chat_id=str(user_turn_reward.chat_id),
        )

        return should_reward, reward_amount, high_value_reward_amount, reward_comment, None, None

    should_get_zero_reward = user_turn_reward.should_get_zero_reward()
    if should_get_zero_reward:
        return _handle_zero_turn_based_reward(user_turn_reward)

    reward_probability = user_turn_reward.get_probability()
    should_reward = random.random() < reward_probability

    # Get reward amount from rule engine
    reward_amount = user_turn_reward.get_amount()
    high_value_reward_amount = user_turn_reward.get_amount(high_value=True)
    reward_comment = user_turn_reward.get_reward_comment()

    # Safety checks.
    if reward_amount < 0:
        reward_amount = 0

    if high_value_reward_amount < max(0, RULE_CONSTANTS["min_ever_high_value_reward_amount"]):
        high_value_reward_amount = max(0, RULE_CONSTANTS["min_ever_high_value_reward_amount"])

    if reward_amount == 0:
        reward_comment = "Better luck next time!"

    log_reward_debug_info(
        should_reward=should_reward,
        reward_amount=reward_amount,
        high_value_reward_amount=high_value_reward_amount,
        **user_turn_reward.to_dict(),
    )

    return (
        should_reward,
        reward_amount,
        high_value_reward_amount,
        reward_comment,
        user_turn_reward.amount_rule,
        user_turn_reward.probability_rule,
    )


def log_reward_debug_info(
    action_type: RewardActionEnum,
    user_id: str,
    user_name: str | None,
    should_reward: bool,
    reward_amount: int,
    probability_rule: RewardProbabilityRule | None,
    amount_rule: RewardAmountRule | None,
    **kwargs: Any,
) -> None:
    """Log reward debug info, and post to Slack if in production."""
    log_dict = {
        "message": f"Reward debug info for user_id={user_id}",
        "action_type": action_type.value.lower(),
        "probability_rule": probability_rule.name if probability_rule else None,
        "amount_rule": amount_rule.name if amount_rule else None,
        "should_reward": should_reward,
        "reward_amount": reward_amount,
        "user_id": user_id,
        "additional_info": kwargs,
    }
    logging.info(json_dumps(log_dict))

    probability_rule_str = f"`{probability_rule.name}`" if probability_rule else "[none]"
    amount_rule_str = f"`{amount_rule.name}`" if amount_rule else "[none]"
    reward_amount_str = f"{reward_amount}"
    if reward_amount == 0:
        reward_amount_str += " :red_circle:"
    user_str = user_name or f"id={user_id}"
    message = (
        f"\nUser: {user_str}\n"
        f"User ID: {user_id}\n"
        f"Type: {action_type.value.lower()}_reward_calculated\n"
        f"Probability Rule: {probability_rule_str}\n"
        f"Amount Rule: {amount_rule_str}\n"
        f"Should Reward: {should_reward}\n"
        f"Reward Amount: {reward_amount_str}\n"
        f"Additional Information:\n```{json_dumps(kwargs, indent=2)}```\n"
    )

    webhook_url = os.environ.get("REWARDS_SLACK_WEBHOOK_URL")
    if settings.ENVIRONMENT != "production" or not webhook_url:
        return
    try:
        post_to_slack_task.delay(message=message, webhook_url=webhook_url)
    except Exception as e:
        log_dict = {"message": "Error posting to Slack", "error": str(e), "user_id": user_id}
        logging.error(json_dumps(log_dict))


def _get_recent_eval_quality_scores(user_id: str, session: Session) -> list[float]:
    return [
        score
        for score in session.exec(
            select(Eval.quality_score)
            .where(Eval.user_id == user_id, Eval.deleted_at.is_(None), Eval.eval_type == EvalType.SELECTION)  # type: ignore
            .order_by(Eval.created_at.desc())  # type: ignore
            .limit(NUM_RECENT_EVALS_FOR_QUALITY_CHECK)
        ).all()
        if score is not None
    ]


def _get_reward_points_summary(user_id: str, session: Session) -> dict[str, int]:
    point_dates: list[tuple[int, datetime, RewardActionEnum]] = session.exec(
        select(PointTransaction.point_delta, PointTransaction.created_at, RewardActionLog.action_type)  # type: ignore
        .join(Reward, PointTransaction.claimed_reward)
        .join(RewardActionLog, Reward.reward_action_logs)
        .where(
            PointTransaction.user_id == user_id,
            PointTransaction.deleted_at.is_(None),  # type: ignore
            PointTransaction.action_type.in_(REWARDABLE_POINT_ACTION_TYPES),  # type: ignore
            PointTransaction.created_at > (datetime.now() - timedelta(days=30)),  # type: ignore
        )
        .order_by(PointTransaction.created_at.desc())  # type: ignore
    ).all()
    results = {
        "points_last_day": 0,
        "points_last_week": 0,
        "points_last_month": 0,
        "points_last_award": 0,
        "referral_points_last_day": 0,
        "referral_points_last_week": 0,
        "referral_points_last_month": 0,
    }
    now = datetime.now(pytz.UTC)
    week_ago = now - timedelta(days=7)
    day_ago = now - timedelta(days=1)
    for point_delta, created_at, action_type in point_dates:
        point_type = "referral_points" if action_type in REFERRAL_REWARD_TYPES else "points"
        results[f"{point_type}_last_month"] += point_delta
        if created_at > week_ago:
            results[f"{point_type}_last_week"] += point_delta
            if created_at > day_ago:
                results[f"{point_type}_last_day"] += point_delta
    if point_dates:
        results["points_last_award"] = point_dates[0][0]

    # Ensure all values are non-negative
    for key, value in results.items():
        results[key] = max(0, value)
    return results


async def generate_bounded_reward(lower_bound: int, upper_bound: int, quality_score: int | None = None) -> int:
    """
    Generate a reward amount between lower and upper bounds, adjusted by quality score.
    Uses a more deterministic approach to ensure higher quality scores always yield higher rewards.

    Args:
        lower_bound (int): Minimum reward amount
        upper_bound (int): Maximum reward amount
        quality_score (int): The feedback quality score (1-5)
    Returns:
        int: The generated reward amount, rounded to nearest 10
    """
    if not quality_score:
        # Default to middle range if no quality score
        mean = (lower_bound + upper_bound) / 2
        std_dev = (upper_bound - lower_bound) / 2
        reward_amount = int(round(random.gauss(mean, std_dev)))
        return max(lower_bound, min(upper_bound, reward_amount))

    range_size = upper_bound - lower_bound

    # Calculate score-based bounds while ensuring they stay within global bounds
    default_multiplier = FEEDBACK_QUALITY_MULTIPLIER[FALLBACK_QUALITY_SCORE]
    quality_multiplier = FEEDBACK_QUALITY_MULTIPLIER.get(quality_score, default_multiplier)
    score_min = lower_bound + (range_size * quality_multiplier * 0.5)
    score_max = lower_bound + (range_size * quality_multiplier * 1.5)

    # Ensure score-based bounds don't exceed global bounds
    score_min = max(lower_bound, min(upper_bound, score_min))
    score_max = max(lower_bound, min(upper_bound, score_max))

    # Add small random variation within the quality score's range
    reward_amount = int(round(random.uniform(score_min, score_max)))
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
    # Get quality score for the comment
    quality_score = FALLBACK_QUALITY_SCORE
    action_type = RewardActionEnum.FEEDBACK

    reward_params = {
        "user_id": user_id,
        "action_type": action_type,
        "quality_score": quality_score,
    }
    with Session(get_engine()) as session:
        reward_params.update(_get_reward_points_summary(user_id, session))

    amount_rule: RewardAmountRule | None = get_matching_rule(get_reward_amount_rules(action_type), reward_params)  # type: ignore
    probability_rule: RewardProbabilityRule | None = get_matching_rule(  # type: ignore
        get_reward_probability_rules(action_type), reward_params
    )

    should_reward = False
    if probability_rule:
        should_reward = random.random() < probability_rule.probability

    min_value = max(FEEDBACK_REWARD_LOWER_BOUND, amount_rule.min_value if amount_rule else FEEDBACK_REWARD_LOWER_BOUND)
    max_value = min(FEEDBACK_REWARD_UPPER_BOUND, amount_rule.max_value if amount_rule else FEEDBACK_REWARD_UPPER_BOUND)
    reward_amount = await generate_bounded_reward(
        lower_bound=min_value,
        upper_bound=max_value,
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

    log_reward_debug_info(
        should_reward=should_reward,
        reward_amount=reward_amount,
        probability_rule=probability_rule,
        amount_rule=amount_rule,
        user_name=None,
        **reward_params,  # type: ignore
    )

    return (
        should_reward,
        reward_amount,
        reward_comment,
        amount_rule,
        probability_rule,
    )


async def qt_eval_reward(user_id: str) -> tuple[bool, int, str, RewardAmountRule | None, RewardProbabilityRule | None]:
    action_type = RewardActionEnum.QT_EVAL
    params = {
        "user_id": user_id,
        "action_type": action_type,
    }
    with Session(get_engine()) as session:
        params.update(_get_reward_points_summary(user_id, session))

    amount_rule: RewardAmountRule | None = get_matching_rule(get_reward_amount_rules(action_type), params)  # type: ignore
    probability_rule: RewardProbabilityRule | None = get_matching_rule(  # type: ignore
        get_reward_probability_rules(action_type), params
    )

    should_reward = False
    if probability_rule:
        should_reward = random.random() < probability_rule.probability

    min_value = amount_rule.min_value if amount_rule else QT_EVAL_REWARD_LOWER_BOUND
    max_value = amount_rule.max_value if amount_rule else QT_EVAL_REWARD_UPPER_BOUND

    reward_amount = await generate_bounded_reward(min_value, max_value)

    reward_comment = f"QT Eval reward: {reward_amount} credits."

    log_reward_debug_info(
        should_reward=should_reward,
        reward_amount=reward_amount,
        probability_rule=None,
        amount_rule=None,
        user_name=None,
        **params,  # type: ignore
    )

    return (
        should_reward,
        reward_amount,
        reward_comment,
        amount_rule,
        probability_rule,
    )


async def sign_up_reward(user_id: str) -> tuple[bool, int, str, RewardAmountRule | None, RewardProbabilityRule | None]:
    action_type = RewardActionEnum.SIGN_UP
    params = {
        "user_id": user_id,
        "action_type": action_type,
        "sign_up_reward_count": await get_user_reward_count_by_action_type(user_id, action_type.name),
    }
    with Session(get_engine()) as session:
        params.update(_get_reward_points_summary(user_id, session))

    amount_rule: RewardAmountRule | None = get_matching_rule(get_reward_amount_rules(action_type), params)  # type: ignore
    probability_rule: RewardProbabilityRule | None = get_matching_rule(  # type: ignore
        get_reward_probability_rules(action_type), params
    )

    if probability_rule:
        should_reward = random.random() < probability_rule.probability
    else:
        should_reward = False

    if amount_rule:
        reward_amount = await generate_bounded_reward(amount_rule.min_value, amount_rule.max_value)
    else:
        reward_amount = 0

    reward_comment = f"Sign up reward: {reward_amount} credits."

    log_reward_debug_info(
        should_reward=should_reward,
        reward_amount=reward_amount,
        probability_rule=probability_rule,
        amount_rule=amount_rule,
        user_name=None,
        **params,  # type: ignore
    )

    return (
        should_reward,
        reward_amount,
        reward_comment,
        amount_rule,
        probability_rule,
    )


async def get_referrer_user_id_for_reward(user_id: str) -> str | None:
    async with AsyncSession(get_async_engine()) as session:
        result = await session.execute(
            select(
                WaitlistedUser.referrer_id,
            )
            .join(User, User.email == WaitlistedUser.email)  # type: ignore
            .join(SpecialInviteCodeClaimLog, SpecialInviteCodeClaimLog.user_id == User.user_id)  # type: ignore
            .join(
                SpecialInviteCode,
                SpecialInviteCode.special_invite_code_id == SpecialInviteCodeClaimLog.special_invite_code_id,  # type: ignore
            )
            .where(User.user_id == user_id, SpecialInviteCode.referral_bonus_eligible.is_(True))  # type: ignore
        )
        referrer_id = result.scalar_one_or_none()
        return referrer_id


async def referral_bonus_reward(
    user_id: str, action_type: RewardActionEnum
) -> tuple[bool, int, str, RewardAmountRule | None, RewardProbabilityRule | None]:
    params = {
        "user_id": user_id,
        "action_type": action_type,
    }
    with Session(get_engine()) as session:
        params.update(_get_reward_points_summary(user_id, session))

    if action_type == RewardActionEnum.REFERRAL_BONUS_REFERRER:
        params["referrer_reward_count"] = await get_user_reward_count_by_action_type(
            user_id, RewardActionEnum.REFERRAL_BONUS_REFERRER.name
        )
    elif action_type == RewardActionEnum.REFERRAL_BONUS_REFERRED_USER:
        params["referred_user_reward_count"] = await get_user_reward_count_by_action_type(
            user_id, RewardActionEnum.REFERRAL_BONUS_REFERRED_USER.name
        )

    amount_rule: RewardAmountRule | None = get_matching_rule(get_reward_amount_rules(action_type), params)  # type: ignore
    probability_rule: RewardProbabilityRule | None = get_matching_rule(  # type: ignore
        get_reward_probability_rules(action_type), params
    )

    if probability_rule:
        should_reward = random.random() < probability_rule.probability
    else:
        should_reward = False

    if amount_rule:
        reward_amount = await generate_bounded_reward(amount_rule.min_value, amount_rule.max_value)
    else:
        reward_amount = 0

    reward_comment = (
        f"Sign up bonus for being referred: {reward_amount} credits."
        if action_type == RewardActionEnum.REFERRAL_BONUS_REFERRED_USER
        else f"Referral bonus for referring a friend: {reward_amount} credits."
    )

    log_reward_debug_info(
        should_reward=should_reward,
        reward_amount=reward_amount,
        probability_rule=probability_rule,
        amount_rule=amount_rule,
        user_name=None,
        **params,  # type: ignore
    )

    return (
        should_reward,
        reward_amount,
        reward_comment,
        amount_rule,
        probability_rule,
    )


@dataclass
class RewardClaimedResponse:
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


async def update_reward_status(reward_id: UUID, user_id: str, new_status: RewardStatusEnum) -> None:
    """
    Update the status of a reward to a non-CLAIMED status.
    Claiming a reward should be done via process_reward_claim.
    """
    async with AsyncSession(get_async_engine()) as session:
        if new_status == RewardStatusEnum.CLAIMED:
            # Use process_reward_claim to claim the reward. This method should not be used to update status to claimed.
            raise ValueError("Cannot update status to claimed")

        restricted_states = []
        if new_status == RewardStatusEnum.REJECTED:
            # Claimed rewards cannot be rejected.
            restricted_states = [RewardStatusEnum.CLAIMED]

        await session.exec(
            update(Reward)
            .returning(Reward.reward_id)  # type: ignore
            .where(
                Reward.reward_id == reward_id,
                Reward.user_id == user_id,
                Reward.deleted_at.is_(None),  # type: ignore
                Reward.status.not_in(restricted_states),  # type: ignore
            )
            .values(status=new_status)
        )
        await session.commit()


def _load_rules_constants() -> Any:
    global RULE_CONSTANTS
    if not RULE_CONSTANTS:
        if not os.path.exists(RULES_PATH):
            log_dict = {
                "message": "Rules file not found",
                "rules_path": str(RULES_PATH),
            }
            logging.error(json_dumps(log_dict))
            return

        with open(RULES_PATH) as f:
            rule_data = yaml.safe_load(f)
            RULE_CONSTANTS = rule_data.get("constants", {})
            return rule_data


def _get_reward_rules(
    rule_class: type[RewardAmountRule] | type[RewardProbabilityRule], rule_type: RewardActionEnum
) -> list[RewardRule]:
    """Get active reward rules of the specified type, sorted by priority."""
    _load_rules_constants()
    with Session(get_engine()) as session:
        rules = session.exec(
            select(rule_class)
            .where(
                rule_class.is_active.is_(True),  # type: ignore
                rule_class.deleted_at.is_(None),  # type: ignore
                rule_class.action_type == rule_type,
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
def get_reward_amount_rules(rule_type: RewardActionEnum = RewardActionEnum.TURN) -> list[RewardRule]:
    return _get_reward_rules(RewardAmountRule, rule_type)


@ttl_cache(ttl=600)  # 10 minute cache
def get_reward_probability_rules(rule_type: RewardActionEnum = RewardActionEnum.TURN) -> list[RewardRule]:
    return _get_reward_rules(RewardProbabilityRule, rule_type)


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


async def get_user_reward_count_by_action_type(user_id: str, action_type: str) -> int:
    """
    Get the count of rewards created for a user for a specific action type.

    Args:
        user_id: The ID of the user
        action_type: The type of reward action to count

    Returns:
        int: The number of rewards created for the given action type
    """
    async with AsyncSession(get_async_engine()) as session:
        query = select(func.count()).where(
            RewardActionLog.user_id == user_id,
            RewardActionLog.action_type == action_type,
            RewardActionLog.deleted_at.is_(None),  # type: ignore
            RewardActionLog.associated_reward_id.is_not(None),  # type: ignore
        )
        result = await session.exec(query)
        return result.one() or 0


async def get_turn_id_from_message_id(message_id: UUID) -> UUID | None:
    """Get turn_id from message_id by querying the chat_messages table."""
    async with AsyncSession(get_async_engine()) as session:
        message = await session.get(ChatMessage, message_id)
        if message:
            return message.turn_id
        return None
