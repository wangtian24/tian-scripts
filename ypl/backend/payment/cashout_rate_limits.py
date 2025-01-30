import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Final, Literal, TypedDict

from fastapi import HTTPException
from sqlalchemy import Column, and_, case, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import true
from sqlalchemy.sql.functions import Function
from sqlmodel import select
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack_with_user_name
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import CapabilityType, UserCapabilityStatus, get_capability_override_details
from ypl.db.payments import PaymentInstrumentFacilitatorEnum
from ypl.db.point_transactions import PointsActionEnum, PointTransaction
from ypl.db.redis import get_upstash_redis_client
from ypl.db.users import User

CASHOUT_KILLSWITCH_KEY = "cashout:killswitch"
# Facilitator specific killswitch, facilitator *SHOULD BE lowercase*
CASHOUT_FACILITATOR_KILLSWITCH_KEY = "cashout:killswitch:facilitator:{facilitator}"

CASHOUT_REQUEST_RATE_LIMIT_ENABLED = True
CASHOUT_REQUEST_RATE_LIMIT: Final[int] = 1  # Only 1 request allowed
CASHOUT_REQUEST_WINDOW_SECONDS: Final[int] = 10  # Within 10 seconds window

MAX_DAILY_CASHOUT_COUNT = 1
MAX_WEEKLY_CASHOUT_COUNT = 2
MAX_MONTHLY_CASHOUT_COUNT = 5
MAX_FIRST_TIME_CASHOUT_CREDITS = 1000

MAX_DAILY_CASHOUT_CREDITS = 10000
MAX_WEEKLY_CASHOUT_CREDITS = 15000
MAX_MONTHLY_CASHOUT_CREDITS = 75000

# Minimum credits that have to be cashed out per transaction.
MINIMUM_CREDITS_PER_CASHOUT = 1000
# Minimum credits that have to be kept after a cashout.
CREDITS_TO_KEEP_AFTER_CASHOUT = 1000

SLACK_WEBHOOK_CASHOUT = settings.SLACK_WEBHOOK_CASHOUT


class PeriodStats(TypedDict):
    count: int
    total: int
    max_count: int
    max_total: int
    period: Literal["daily", "weekly", "monthly"]


class RawResults(TypedDict):
    daily_cashouts: int
    daily_reversals: int
    daily_total: int
    weekly_cashouts: int
    weekly_reversals: int
    weekly_total: int
    monthly_cashouts: int
    monthly_reversals: int
    monthly_total: int
    total_count: int


class CashoutLimitType(Enum):
    TRANSACTION_COUNT = "transaction count"
    CREDIT_AMOUNT = "credit amount"
    FIRST_TIME = "first time credit"
    RATE_LIMIT = "rate limit"
    DISABLED = "disabled"
    INSUFFICIENT_CREDIT_BALANCE = "insufficient credit balance"
    MINIMUM_CREDITS_PER_CASHOUT = "minimum credits per cashout"


class CashoutLimitError(HTTPException):
    """Custom exception for cashout limit violations."""

    def __init__(
        self,
        period: str,
        limit_type: CashoutLimitType,
        current_value: int,
        max_value: int,
        user_id: str,
        min_value: int = 0,
    ):
        self.period = period
        self.limit_type = limit_type
        self.current_value = current_value
        self.max_value = max_value
        self.user_id = user_id
        self.min_value = min_value

        # Client will show these messages to the user.
        # Keep it ambiguous to avoid leaking information about the limit.
        if limit_type == CashoutLimitType.RATE_LIMIT:
            detail = f"Please wait {max_value} seconds before submitting another cash out request"
        elif limit_type == CashoutLimitType.DISABLED:
            detail = "Cash out is disabled for your account.\nPlease contact support if you think this is an error."
        elif limit_type == CashoutLimitType.INSUFFICIENT_CREDIT_BALANCE:
            detail = "You have insufficient credits to cash out"
        elif limit_type == CashoutLimitType.MINIMUM_CREDITS_PER_CASHOUT:
            detail = f"You must cash out at least {min_value} credits"
        else:
            detail = f"You have reached the {period} cash out limit.\nPlease try again later."

        super().__init__(status_code=429 if limit_type == CashoutLimitType.RATE_LIMIT else 400, detail=detail)


def log_cashout_limit_error(
    error: CashoutLimitError,
    is_precheck: bool,
) -> None:
    # Log the error
    log_dict = {
        "message": f":x: Failure - {error.period.capitalize()} cashout limit reached",
        "user_id": error.user_id,
        "limit_type": error.limit_type.value,
        "current_value": error.current_value,
        "min_value": error.min_value,
        "max_value": error.max_value,
        "period": error.period,
        "error_message": error.detail,
        "is_precheck": is_precheck,
    }
    logging.warning(json_dumps(log_dict))
    # Don't post to slack if the exception was raised during a pre-cashout validation.
    if not is_precheck:
        asyncio.create_task(post_to_slack_with_user_name(error.user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))


def build_time_window_stats(
    created_at_col: Column, point_delta_col: Column, transaction_id_col: Column, cutoff_time: datetime
) -> list[Function]:
    """Helper function to build stats for a time window.
    Returns:
    - Count of cashout initiated transactions (negative point deltas)
    - Count of reversal transactions (positive point deltas)
    - Sum of all point deltas
    """

    # This is a common condition that will be used in multiple queries
    # So depending on whether daily, weekly or monthly, we can use the same condition
    # and only the value of cutoff_time will change. This is passed from the caller
    time_window_condition = created_at_col.isnot(None) & (created_at_col >= cutoff_time)

    return [
        # Count of cashout initiated transactions
        # negative point deltas indicate cashout initiated transactions
        func.count(
            case(
                {true(): transaction_id_col},
                value=time_window_condition & (point_delta_col < 0),
                else_=None,
            )
        ),
        # Count of reversal transactions
        # positive point deltas indicate reversal transactions
        func.count(
            case(
                {true(): transaction_id_col},
                value=time_window_condition & (point_delta_col > 0),
                else_=None,
            )
        ),
        # Sum of all point deltas
        # we don't need to worry about reversals as we are interested in net total credits cashed out
        func.abs(
            func.coalesce(
                func.sum(
                    case(
                        {true(): point_delta_col},
                        value=time_window_condition,
                        else_=0,
                    )
                ),
                0,
            )
        ),
    ]


async def get_cashout_stats(
    session: AsyncSession, user_id: str, time_windows: dict[str, datetime]
) -> tuple[list[PeriodStats] | None, bool]:
    """
    Get cashout statistics for a user across different time windows.
    Returns a tuple of (list of stats per period, is_first_time).
    If no previous cashouts are found, returns (None, True).
    """

    # build the sql query to get the stats for each time window
    base_filters = [
        PointTransaction.user_id == user_id,
        PointTransaction.action_type.in_([PointsActionEnum.CASHOUT, PointsActionEnum.CASHOUT_REVERSED]),  # type: ignore[attr-defined]
        PointTransaction.deleted_at.is_(None),  # type: ignore[union-attr]
    ]

    select_columns = []
    for window_name, cutoff_time in time_windows.items():
        cashouts, reversals, total = build_time_window_stats(
            PointTransaction.created_at,  # type: ignore[arg-type]
            PointTransaction.point_delta,  # type: ignore[arg-type]
            PointTransaction.transaction_id,  # type: ignore[arg-type]
            cutoff_time,
        )
        select_columns.extend(
            [
                cashouts.label(f"{window_name}_cashouts"),
                reversals.label(f"{window_name}_reversals"),
                total.label(f"{window_name}_total"),
            ]
        )

    select_columns.append(func.count().label("total_count"))

    stats = await session.execute(select(*select_columns).where(and_(*base_filters)))
    result = stats.first()

    if not result or result.total_count == 0:
        log_dict = {
            "message": "No previous cashouts found",
            "user_id": user_id,
        }
        logging.info(json_dumps(log_dict))

        return None, True

    # Unpack all values from the result
    (
        daily_cashouts,
        daily_reversals,
        daily_total,
        weekly_cashouts,
        weekly_reversals,
        weekly_total,
        monthly_cashouts,
        monthly_reversals,
        monthly_total,
        total_count,
    ) = result

    # build the statistics for the user for each time window
    # this is what would be compared against the limits
    period_stats: list[PeriodStats] = [
        {
            "count": daily_cashouts - daily_reversals,  # Net daily successful cashouts
            "total": daily_total,
            "max_count": MAX_DAILY_CASHOUT_COUNT,
            "max_total": MAX_DAILY_CASHOUT_CREDITS,
            "period": "daily",
        },
        {
            "count": weekly_cashouts - weekly_reversals,  # Net weekly successful cashouts
            "total": weekly_total,
            "max_count": MAX_WEEKLY_CASHOUT_COUNT,
            "max_total": MAX_WEEKLY_CASHOUT_CREDITS,
            "period": "weekly",
        },
        {
            "count": monthly_cashouts - monthly_reversals,  # Net monthly successful cashouts
            "total": monthly_total,
            "max_count": MAX_MONTHLY_CASHOUT_COUNT,
            "max_total": MAX_MONTHLY_CASHOUT_CREDITS,
            "period": "monthly",
        },
    ]

    log_dict = {
        "message": "Cashout stats for limits validation",
        "user_id": user_id,
        "stats": str([dict(stat) for stat in period_stats]),
    }
    logging.info(json_dumps(log_dict))

    return period_stats, False


async def validate_and_return_first_time_cashout_limits(
    user_id: str, credits_to_cashout: int, override_config: dict | None = None
) -> int:
    """
    Validate if this is user's first cashout and check against first-time limits.

    Args:
        user_id: The ID of the user to check
        credits_to_cashout: The number of credits to cash out. Set to 0 if the call is made in a pre-cashout validation.
        override_config: The override config for the user, if any.

    Returns:
        int: The first time cashout limit for the user.
    """
    first_time_limit = (
        override_config.get("first_time_limit", MAX_FIRST_TIME_CASHOUT_CREDITS)
        if override_config
        else MAX_FIRST_TIME_CASHOUT_CREDITS
    )

    log_dict = {
        "message": "First-time cashout limit validation",
        "user_id": user_id,
        "credits_to_cashout": credits_to_cashout,
        "first_time_limit": first_time_limit,
        "is_override": first_time_limit != MAX_FIRST_TIME_CASHOUT_CREDITS,
    }
    logging.info(json_dumps(log_dict))

    if credits_to_cashout > first_time_limit:
        raise CashoutLimitError(
            period="first time",
            limit_type=CashoutLimitType.FIRST_TIME,
            current_value=credits_to_cashout,
            max_value=first_time_limit,
            user_id=user_id,
        )

    return int(first_time_limit)


async def validate_period_limits_and_return_limiting_credits(
    user_id: str, credits_to_cashout: int, period_stats: list[PeriodStats], override_config: dict | None = None
) -> int:
    """
    Validate cashout limits for each time period.

    Args:
        user_id: The ID of the user to check
        credits_to_cashout: The number of credits to cash out. Set to 0 if the call is made in a pre-cashout validation.
        period_stats: The stats for the user for each time period.
        override_config: The override config for the user, if any.

    Returns:
        int: The maximum amount that the user can cashout, without exceeding any limits.
            This method doesn't account for the credits that the user currently has,
            it only checks against the limits.
    """
    # Get override limits if available
    daily_count = (
        override_config.get("daily_count", MAX_DAILY_CASHOUT_COUNT) if override_config else MAX_DAILY_CASHOUT_COUNT
    )
    weekly_count = (
        override_config.get("weekly_count", MAX_WEEKLY_CASHOUT_COUNT) if override_config else MAX_WEEKLY_CASHOUT_COUNT
    )
    monthly_count = (
        override_config.get("monthly_count", MAX_MONTHLY_CASHOUT_COUNT)
        if override_config
        else MAX_MONTHLY_CASHOUT_COUNT
    )

    daily_credits = (
        override_config.get("daily_credits", MAX_DAILY_CASHOUT_CREDITS)
        if override_config
        else MAX_DAILY_CASHOUT_CREDITS
    )
    weekly_credits = (
        override_config.get("weekly_credits", MAX_WEEKLY_CASHOUT_CREDITS)
        if override_config
        else MAX_WEEKLY_CASHOUT_CREDITS
    )
    monthly_credits = (
        override_config.get("monthly_credits", MAX_MONTHLY_CASHOUT_CREDITS)
        if override_config
        else MAX_MONTHLY_CASHOUT_CREDITS
    )

    # Update the period stats with override values
    for limit in period_stats:
        if limit["period"] == "daily":
            limit["max_count"] = daily_count
            limit["max_total"] = daily_credits
        elif limit["period"] == "weekly":
            limit["max_count"] = weekly_count
            limit["max_total"] = weekly_credits
        elif limit["period"] == "monthly":
            limit["max_count"] = monthly_count
            limit["max_total"] = monthly_credits

    # List of credits that will limit the user's cashout for each period.
    limiting_credits = []
    # for each limit defined for the user, check if the current value has exceeded the max value
    for limit in period_stats:
        if limit["count"] >= limit["max_count"]:
            raise CashoutLimitError(
                period=limit["period"],
                limit_type=CashoutLimitType.TRANSACTION_COUNT,
                current_value=limit["count"],
                max_value=limit["max_count"],
                user_id=user_id,
            )

        if limit["total"] + credits_to_cashout > limit["max_total"]:
            raise CashoutLimitError(
                period=limit["period"],
                limit_type=CashoutLimitType.CREDIT_AMOUNT,
                current_value=limit["total"] + credits_to_cashout,
                max_value=limit["max_total"],
                user_id=user_id,
            )

        limiting_credits.append(limit["max_total"] - limit["total"])

    return min(limiting_credits)


async def check_request_rate_limit(user_id: str) -> None:
    """Check if the user has exceeded the rate limit for cashout requests."""
    if settings.ENVIRONMENT != "production" or not CASHOUT_REQUEST_RATE_LIMIT_ENABLED:
        return

    redis = await get_upstash_redis_client()
    bucket_key = f"cashout_request:{user_id}"

    current_count_str = await redis.get(bucket_key)
    if current_count_str is not None and int(current_count_str) >= CASHOUT_REQUEST_RATE_LIMIT:
        ttl = await redis.ttl(bucket_key)
        wait_seconds = ttl if ttl > 0 else CASHOUT_REQUEST_WINDOW_SECONDS
        raise CashoutLimitError(
            period="request",
            limit_type=CashoutLimitType.RATE_LIMIT,
            current_value=int(current_count_str),
            max_value=wait_seconds,
            user_id=user_id,
        )

    current_count = await redis.incr(bucket_key)
    if current_count == 1:
        await redis.expire(bucket_key, CASHOUT_REQUEST_WINDOW_SECONDS)


async def _get_credits_balance(session: AsyncSession, user_id: str) -> tuple[int, bool]:
    """
    Get the user's current credits balance.
    """
    query = select(User.points, User.email).where(
        User.user_id == user_id,
        User.deleted_at.is_(None),  # type: ignore
    )
    result = await session.execute(query)
    points, email = result.one()
    return points, email.endswith("@yupp.ai")


@dataclass
class CashoutUserInfo:
    credits_balance: int
    credits_available_for_cashout: int
    minimum_credits_per_cashout: int = MINIMUM_CREDITS_PER_CASHOUT


async def validate_and_return_cashout_user_limits(user_id: str, credits_to_cashout: int = 0) -> CashoutUserInfo:
    """
    Validate the cashout request, and enforce limits.
    Checks both count of cashouts and total credits cashed out for daily, weekly and monthly periods.
    Raises CashoutLimitError if any limit is exceeded.

    Args:
        user_id: The ID of the user to check
        credits_to_cashout: The number of credits to cash out. Set to 0 if the call is made in a pre-cashout validation.

    Returns:
        CashoutUserLimits: The limits for the user if they did not exceed any limits.
    """

    now = datetime.utcnow()
    time_windows = {
        "daily": now - timedelta(days=1),
        "weekly": now - timedelta(days=7),
        "monthly": now - timedelta(days=30),
    }

    async with get_async_session() as session:
        credits_balance, is_yuppster = await _get_credits_balance(session, user_id)
        if credits_to_cashout != 0 and credits_to_cashout < MINIMUM_CREDITS_PER_CASHOUT:
            # User is trying to cashout less than the minimum amount.
            raise CashoutLimitError(
                period="request",
                limit_type=CashoutLimitType.MINIMUM_CREDITS_PER_CASHOUT,
                current_value=credits_to_cashout,
                min_value=MINIMUM_CREDITS_PER_CASHOUT,
                max_value=0,
                user_id=user_id,
            )
        total_credits_available_for_cashout = (
            credits_balance - CREDITS_TO_KEEP_AFTER_CASHOUT
            if credits_balance > CREDITS_TO_KEEP_AFTER_CASHOUT + MINIMUM_CREDITS_PER_CASHOUT
            else 0
        )
        if total_credits_available_for_cashout < MINIMUM_CREDITS_PER_CASHOUT:
            raise CashoutLimitError(
                period="request",
                limit_type=CashoutLimitType.INSUFFICIENT_CREDIT_BALANCE,
                current_value=total_credits_available_for_cashout,
                max_value=0,
                user_id=user_id,
            )
        cashout_user_info = CashoutUserInfo(
            credits_balance=credits_balance,
            # This value will be updated later in the method, based on the limits.
            credits_available_for_cashout=total_credits_available_for_cashout,
            minimum_credits_per_cashout=MINIMUM_CREDITS_PER_CASHOUT,
        )
        if settings.ENVIRONMENT != "production":
            return cashout_user_info

        capability_override_details = await get_capability_override_details(session, user_id, CapabilityType.CASHOUT)
        log_dict = {
            "message": "Cashout override details",
            "user_id": user_id,
            "override_status": capability_override_details["status"] if capability_override_details else "None",
            "override_config": str(capability_override_details["override_config"])
            if capability_override_details
            else "None",
        }
        logging.info(json_dumps(log_dict))
        if capability_override_details and capability_override_details["status"] == UserCapabilityStatus.DISABLED:
            log_dict = {
                "message": "Cashout is disabled for user",
                "user_id": user_id,
            }
            logging.info(json_dumps(log_dict))

            raise CashoutLimitError(
                period="request",
                limit_type=CashoutLimitType.DISABLED,
                current_value=0,
                max_value=0,
                user_id=user_id,
            )

        # If the user is a yuppster, we shouldn't check rest of the limits.
        # Yuppsters short-circuit rest of the limits.
        # Ensure that we do the capability override checks before, just in case.
        if is_yuppster:
            return cashout_user_info

        period_stats, is_first_time = await get_cashout_stats(session, user_id, time_windows)

        override_config = (
            capability_override_details["override_config"]
            if capability_override_details and capability_override_details["status"] == UserCapabilityStatus.ENABLED
            else None
        )

        log_dict = {
            "message": "Cashout override details",
            "user_id": user_id,
            "override_config": str(override_config) if override_config else "None",
        }
        logging.info(json_dumps(log_dict))

        if is_first_time:
            first_time_limit = await validate_and_return_first_time_cashout_limits(
                user_id, credits_to_cashout, override_config
            )
            cashout_user_info.credits_available_for_cashout = min(first_time_limit, total_credits_available_for_cashout)
            return cashout_user_info

        assert period_stats is not None
        limiting_credits = await validate_period_limits_and_return_limiting_credits(
            user_id, credits_to_cashout, period_stats, override_config
        )
        cashout_user_info.credits_available_for_cashout = min(limiting_credits, total_credits_available_for_cashout)
        return cashout_user_info


class CashoutKillswitchError(Exception):
    """Raise this error when cashout killswitch is enabled."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def _log_killswitch_error(
    message: str, user_id: str, facilitator: PaymentInstrumentFacilitatorEnum | None = None
) -> None:
    log_dict = {
        "message": message,
        "user_id": user_id,
    }
    if facilitator:
        log_dict["facilitator"] = facilitator.name
    logging.warning(json_dumps(log_dict))
    asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))


async def check_cashout_killswitch(facilitator: PaymentInstrumentFacilitatorEnum, user_id: str) -> None:
    """Check if cashout is enabled, globally or for the given facilitator."""

    if settings.ENVIRONMENT != "production":
        return

    await check_global_cashout_killswitch(user_id)
    await check_facilitator_cashout_killswitch(facilitator, user_id)


async def check_global_cashout_killswitch(user_id: str) -> None:
    """Check if there's a global cashout killswitch."""
    redis_client = await get_upstash_redis_client()
    if await redis_client.get(CASHOUT_KILLSWITCH_KEY):
        _log_killswitch_error(":x: Failure - Cashout is currently disabled", user_id)
        raise CashoutKillswitchError("Cash out is currently disabled")


async def check_facilitator_cashout_killswitch(facilitator: PaymentInstrumentFacilitatorEnum, user_id: str) -> None:
    """Check if there's a cashout killswitch for the given facilitator."""
    redis_client = await get_upstash_redis_client()
    if await redis_client.get(CASHOUT_FACILITATOR_KILLSWITCH_KEY.format(facilitator=facilitator.value)):
        _log_killswitch_error(
            f":x: Failure - Cashout is currently disabled for {facilitator.name}", user_id, facilitator
        )
        raise CashoutKillswitchError("Cash out is currently disabled for this payment method")
