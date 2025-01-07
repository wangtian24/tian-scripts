import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal, TypedDict

from fastapi import HTTPException
from sqlalchemy import Column, and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import true
from sqlalchemy.sql.functions import Function
from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.point_transactions import PointsActionEnum, PointTransaction

# Constants for rate limiting
MAX_DAILY_CASHOUT_COUNT = 5
MAX_WEEKLY_CASHOUT_COUNT = 15
MAX_MONTHLY_CASHOUT_COUNT = 50
MAX_FIRST_TIME_CASHOUT_CREDITS = 25000
MAX_DAILY_CASHOUT_CREDITS = 50000
MAX_WEEKLY_CASHOUT_CREDITS = 150000
MAX_MONTHLY_CASHOUT_CREDITS = 500000


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


class CashoutLimitError(HTTPException):
    """Custom exception for cashout limit violations."""

    def __init__(self, period: str, limit_type: CashoutLimitType, current_value: int, max_value: int, user_id: str):
        self.period = period
        self.limit_type = limit_type
        self.current_value = current_value
        self.max_value = max_value
        self.user_id = user_id

        detail = f"{period.capitalize()} cashout {limit_type.value} limit of {max_value} reached"
        super().__init__(status_code=400, detail=detail)

        # Log the error
        log_dict = {
            "message": f"{period.capitalize()} cashout limit reached",
            "user_id": user_id,
            "limit_type": limit_type.value,
            "current_value": current_value,
            "max_value": max_value,
            "period": period,
        }
        logging.warning(json_dumps(log_dict))


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

    if not result:
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


async def validate_first_time_cashout(user_id: str, credits_to_cashout: int) -> None:
    """Validate if this is user's first cashout and check against first-time limits."""

    log_dict = {
        "message": "First-time cashout limit validation",
        "user_id": user_id,
        "credits_to_cashout": credits_to_cashout,
        "first_time_limit": MAX_FIRST_TIME_CASHOUT_CREDITS,
    }
    logging.info(json_dumps(log_dict))

    if credits_to_cashout > MAX_FIRST_TIME_CASHOUT_CREDITS:
        raise CashoutLimitError(
            period="first time",
            limit_type=CashoutLimitType.FIRST_TIME,
            current_value=credits_to_cashout,
            max_value=MAX_FIRST_TIME_CASHOUT_CREDITS,
            user_id=user_id,
        )


async def validate_period_limits(user_id: str, credits_to_cashout: int, period_stats: list[PeriodStats]) -> None:
    """Validate cashout limits for each time period."""

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


async def validate_user_cashout_limits(user_id: str, credits_to_cashout: int) -> None:
    """
    Validate that the user hasn't exceeded their cashout limits.
    Checks both count of cashouts and total credits cashed out for daily, weekly and monthly periods.
    Raises HTTPException if any limit is exceeded.
    """
    now = datetime.utcnow()
    time_windows = {
        "daily": now - timedelta(days=1),
        "weekly": now - timedelta(days=7),
        "monthly": now - timedelta(days=30),
    }

    async with get_async_session() as session:
        period_stats, is_first_time = await get_cashout_stats(session, user_id, time_windows)

        if is_first_time:
            await validate_first_time_cashout(user_id, credits_to_cashout)
            return

        assert period_stats is not None
        await validate_period_limits(user_id, credits_to_cashout, period_stats)
