import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Final, Literal, TypedDict

from fastapi import HTTPException
from sqlalchemy import Column, and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import true
from sqlalchemy.sql.functions import Function
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack_with_user_name
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import CapabilityType, UserCapabilityStatus, get_capability_override_details
from ypl.db.payments import PaymentInstrumentFacilitatorEnum
from ypl.db.point_transactions import PointsActionEnum, PointTransaction
from ypl.db.redis import get_upstash_redis_client

CASHOUT_KILLSWITCH_KEY = "cashout:killswitch"
# Facilitator specific killswitch, facilitator *SHOULD BE lowercase*
CASHOUT_FACILITATOR_KILLSWITCH_KEY = "cashout:killswitch:facilitator:{facilitator}"

CASHOUT_REQUEST_RATE_LIMIT_ENABLED = True
CASHOUT_REQUEST_RATE_LIMIT: Final[int] = 1  # Only 1 request allowed
CASHOUT_REQUEST_WINDOW_SECONDS: Final[int] = 10  # Within 10 seconds window


MAX_DAILY_CASHOUT_COUNT = 1
MAX_WEEKLY_CASHOUT_COUNT = 1
MAX_MONTHLY_CASHOUT_COUNT = 5
# TODO(arawind, ENG-1708): Fix this value after the UI knows whether the user is first time or not.
# Keep this value in sync with the MAX_CREDITS_FOR_CASHOUT value in lib/credits.ts.
# Keep this value in sync with daily_points_limit in data/reward_rules.yml.
MAX_FIRST_TIME_CASHOUT_CREDITS = 15000

MAX_DAILY_CASHOUT_CREDITS = 15000
MAX_WEEKLY_CASHOUT_CREDITS = 15000
MAX_MONTHLY_CASHOUT_CREDITS = 75000

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


class CashoutLimitError(HTTPException):
    """Custom exception for cashout limit violations."""

    def __init__(self, period: str, limit_type: CashoutLimitType, current_value: int, max_value: int, user_id: str):
        self.period = period
        self.limit_type = limit_type
        self.current_value = current_value
        self.max_value = max_value
        self.user_id = user_id

        # Client will show this message to the user.
        # Keep it ambiguous to avoid leaking information about the limit.
        if limit_type == CashoutLimitType.RATE_LIMIT:
            detail = f"Please wait {max_value} seconds before submitting another cashout request"
        else:
            detail = f"You have reached the {period} cashout limit. Please try again later."

        super().__init__(status_code=429 if limit_type == CashoutLimitType.RATE_LIMIT else 400, detail=detail)

        # Log the error
        log_dict = {
            "message": f"{period.capitalize()} cashout limit reached",
            "user_id": user_id,
            "limit_type": limit_type.value,
            "current_value": current_value,
            "max_value": max_value,
            "period": period,
            "error_message": detail,
        }
        logging.warning(json_dumps(log_dict))
        asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))


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


async def validate_first_time_cashout(
    user_id: str, credits_to_cashout: int, override_config: dict | None = None
) -> None:
    """Validate if this is user's first cashout and check against first-time limits."""
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


async def validate_period_limits(
    user_id: str, credits_to_cashout: int, period_stats: list[PeriodStats], override_config: dict | None = None
) -> None:
    """Validate cashout limits for each time period."""
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


async def check_request_rate_limit(user_id: str) -> None:
    """Check if the user has exceeded the rate limit for cashout requests."""
    if settings.ENVIRONMENT != "production":
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


async def validate_user_cashout_limits(user_id: str, credits_to_cashout: int) -> None:
    """
    Validate that the user hasn't exceeded their cashout limits.
    Checks both count of cashouts and total credits cashed out for daily, weekly and monthly periods.
    Also checks the request rate limit.
    Raises HTTPException if any limit is exceeded.
    """

    if settings.ENVIRONMENT != "production":
        return

    if CASHOUT_REQUEST_RATE_LIMIT_ENABLED:
        await check_request_rate_limit(user_id)

    now = datetime.utcnow()
    time_windows = {
        "daily": now - timedelta(days=1),
        "weekly": now - timedelta(days=7),
        "monthly": now - timedelta(days=30),
    }

    async with get_async_session() as session:
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
                limit_type=CashoutLimitType.RATE_LIMIT,
                current_value=0,
                max_value=0,
                user_id=user_id,
            )

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
            await validate_first_time_cashout(user_id, credits_to_cashout, override_config)
            return

        assert period_stats is not None
        await validate_period_limits(user_id, credits_to_cashout, period_stats, override_config)


class CashoutKillswitchError(Exception):
    """Raise this error when cashout killswitch is enabled."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def _log_killswitch_error(message: str, facilitator: PaymentInstrumentFacilitatorEnum, user_id: str) -> None:
    log_dict = {
        "message": message,
        "user_id": user_id,
        "facilitator": facilitator.name,
    }
    logging.warning(json_dumps(log_dict))
    asyncio.create_task(post_to_slack_with_user_name(user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))


async def check_cashout_killswitch(facilitator: PaymentInstrumentFacilitatorEnum, user_id: str) -> None:
    """Check if cashout is enabled, globally or for the given facilitator."""

    if settings.ENVIRONMENT != "production":
        return

    redis_client = await get_upstash_redis_client()

    if await redis_client.get(CASHOUT_KILLSWITCH_KEY):
        _log_killswitch_error("Cashout is currently disabled", facilitator, user_id)
        raise CashoutKillswitchError("Cashout is currently disabled")

    if await redis_client.get(CASHOUT_FACILITATOR_KILLSWITCH_KEY.format(facilitator=facilitator.value)):
        _log_killswitch_error(f"Cashout is currently disabled for {facilitator.name}", facilitator, user_id)
        raise CashoutKillswitchError("Cashout is currently disabled for this payment method")
