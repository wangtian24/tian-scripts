import csv
import logging
import os
import time
from base64 import b64encode
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Final

import requests
from requests.exceptions import RequestException
from typing_extensions import TypedDict
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.routes.v1.credit import CREDITS_TO_USD_RATE
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import fetch_user_names


class ChartInfo(TypedDict):
    id: str
    description: str
    series_number: int


AMPLITUDE_CHARTS: Final[dict[str, ChartInfo]] = {
    "users": {"id": "ruptpxku", "description": "active users", "series_number": 0},
    "users_any_chat": {"id": "ruptpxku", "description": "users any chat", "series_number": 1},
    "users_start_chat": {"id": "ruptpxku", "description": "users start chat", "series_number": 2},
    "users_pref": {"id": "ruptpxku", "description": "users pref", "series_number": 3},
    "users_mof": {"id": "ruptpxku", "description": "users mof", "series_number": 4},
    "conversations": {"id": "ekszggrp", "description": "conversations", "series_number": 0},
    "follow_up": {"id": "ekszggrp", "description": "follow up", "series_number": 1},
    "show_more": {"id": "ekszggrp", "description": "show more", "series_number": 2},
    "models_used": {"id": "czeomci2", "description": "models used", "series_number": 0},
    "streaming_complete": {"id": "w9grx38g", "description": "streaming complete", "series_number": 0},
    "streaming_errors": {"id": "w9grx38g", "description": "streaming errors", "series_number": 1},
    "streaming_stopped": {"id": "w9grx38g", "description": "streaming stopped", "series_number": 2},
    "qt_refusals": {"id": "cnshoe8v", "description": "QT refusals", "series_number": 0},
    "qt_latency": {"id": "cnshoe8v", "description": "QT latency", "series_number": 1},
    "qt_evals": {"id": "cnshoe8v", "description": "QT evals", "series_number": 2},
    "qt_thumbs_up": {"id": "cnshoe8v", "description": "QT thumbs up", "series_number": 3},
    "qt_thumbs_down": {"id": "cnshoe8v", "description": "QT thumbs down", "series_number": 4},
    "feedbacks_pref": {"id": "784miua4", "description": "feedbacks pref", "series_number": 0},
    "feedbacks_mof": {"id": "784miua4", "description": "feedbacks mof", "series_number": 1},
    "feedbacks_af": {"id": "784miua4", "description": "feedbacks af", "series_number": 2},
    "credits_total": {"id": "7ukm5tc7", "description": "credits total", "series_number": 0},
    "credits_af": {"id": "7ukm5tc7", "description": "credits af", "series_number": 1},
    "credits_pref": {"id": "7ukm5tc7", "description": "credits pref", "series_number": 2},
    "credits_qt": {"id": "7ukm5tc7", "description": "credits qt", "series_number": 3},
    "cashout_users": {"id": "m33ch3su", "description": "cashout users", "series_number": 0},
    "cashout_count": {"id": "q4615u1m", "description": "cashout count", "series_number": 0},
    "cashout_amount": {"id": "0igijx4l", "description": "cashout credits initiated", "series_number": 0},
    "waitlist_sign_up": {"id": "mrdablao", "description": "waitlist sign up", "series_number": 0},
    "waitlist_sign_up_no_google": {
        "id": "mrdablao",
        "description": "waitlist sign up (no Google account)",
        "series_number": 1,
    },
    "sic_submitted_valid": {"id": "mrdablao", "description": "valid SIC submitted", "series_number": 2},
    "sic_submitted_total": {
        "id": "mrdablao",
        "description": "SIC submitted (valid + invalid)",
        "series_number": 3,
    },
    "sic_dialog_open": {"id": "mrdablao", "description": "viewed invite dialog", "series_number": 4},
    "sic_post_signup_chat": {"id": "1wds3nom", "description": "submit SIC -> start chat funnel", "series_number": 1},
    "sic_post_signup_pref": {"id": "1wds3nom", "description": "submit SIC -> pref funnel", "series_number": 2},
}


class CohortInfo(TypedDict):
    id: str
    description: str


class AmplitudeError(Exception):
    """Base exception for Amplitude-related errors."""

    pass


AMPLITUDE_COHORTS: Final[dict[str, CohortInfo]] = {
    "daily_active_guests": {"id": "scljjdk9", "description": "Daily Active Guests"},
    "weekly_active_guests": {"id": "wmr7jmrw", "description": "Weekly Active Guests"},
    "daily_transacting_guests": {"id": "aeja2n3d", "description": "Daily Transacting Guests"},
}


class AmplitudeTimeoutError(AmplitudeError):
    """Raised when Amplitude requests timeout."""

    pass


def fetch_chart_data(chart_id: str, auth: str, start_date: datetime, end_date: datetime) -> dict[str, Any]:
    """Fetch data from a specific Amplitude chart.

    Args:
        chart_id: The ID of the Amplitude chart.
        auth: Basic auth header string.
        start_date: Start date for the query.
        end_date: End date for the query.

    Returns:
        Dict containing the chart data.

    Raises:
        RequestException: If the API request fails.
        ValueError: If the response format is invalid.
    """
    url = f"https://amplitude.com/api/3/chart/{chart_id}/query"

    params = {
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
    }

    try:
        response = requests.get(
            url, params=params, headers={"Authorization": f"Basic {auth}", "Accept": "application/json"}
        )
        response.raise_for_status()
        return dict(response.json())
    except RequestException as e:
        logging.error(f"Failed to fetch data from Amplitude: {e}")
        raise


def fetch_cohort_data(cohort_id: str, auth: str, start_date: datetime, end_date: datetime) -> dict[str, Any]:
    """Fetch data from a specific Amplitude cohort."""
    url = f"https://amplitude.com/api/5/cohorts/request/{cohort_id}?props=1"

    try:
        response = requests.get(url, headers={"Authorization": f"Basic {auth}", "Accept": "application/json"})
        response.raise_for_status()
        request_id = response.json()["request_id"]

        timeout = datetime.now() + timedelta(minutes=10)
        while datetime.now() < timeout:
            status_url = f"https://amplitude.com/api/5/cohorts/request-status/{request_id}"
            response = requests.get(status_url, headers={"Host": "amplitude.com", "Authorization": f"Basic {auth}"})
            response.raise_for_status()
            status = response.json()["async_status"]
            if status == "JOB COMPLETED":
                cohort_url = f"https://amplitude.com/api/5/cohorts/request/{request_id}/file"
                response = requests.get(cohort_url, headers={"Authorization": f"Basic {auth}"})
                response.raise_for_status()

                csv_data = StringIO(response.text)
                reader = csv.DictReader(csv_data)
                user_ids = [row["user_id"].strip('"') for row in reader]  # Remove quotes from user_ids

                return {"ids": user_ids}

            time.sleep(60)

        raise ValueError(f"Timeout waiting for cohort data. Last status: {status}")

    except RequestException as e:
        logging.error(f"Failed to fetch data from Amplitude cohort: {e}")
        raise


async def post_data_from_charts(auth: str, start_date: datetime, end_date: datetime) -> None:
    """Fetch and post data from multiple Amplitude charts.

    Args:
        auth: Basic auth header string.
        start_date: Start date for the query.
        end_date: End date for the query.

    Raises:
        RequestException: If API requests fail.
    """
    message = f"*Amplitude Metrics for {start_date.date()}*\n"
    log_dict: dict[str, int] = {}

    try:
        charts_items = list(AMPLITUDE_CHARTS.items())
        metrics: dict[str, int] = {}

        for metric, chart_info in charts_items:
            try:
                data = fetch_chart_data(chart_id=chart_info["id"], auth=auth, start_date=start_date, end_date=end_date)
                logging.info(json_dumps(data.get("data", {})))
                # Check if the data type is from a funnel graph
                if isinstance(data.get("data"), list) and "cumulativeRaw" in data["data"][0]:
                    series_data = data["data"][0]["cumulativeRaw"][chart_info["series_number"]]
                    metrics[metric] = series_data
                    log_dict[chart_info["description"]] = series_data
                else:  # else assume that it is a time series.
                    series_data = data["data"]["series"][chart_info["series_number"]]
                    yesterdays_value = int(series_data[-2]["value"])
                    metrics[metric] = yesterdays_value
                    log_dict[chart_info["description"]] = yesterdays_value
            except (KeyError, IndexError, RequestException) as e:
                logging.error(f"Failed to process data for {metric}: {e}")
                metrics[metric] = 0

        message = f"*Amplitude Metrics for {start_date.date()}*\n"

        # a. Users section
        message += (
            f"a. Users: {metrics.get('users', 0)} active, "
            f"{metrics.get('users_any_chat', 0)} had a new or follow up chat, "
            f"{metrics.get('users_start_chat', 0)} started a new chat, "
            f"{metrics.get('users_pref', 0)} did PREF, "
            f"{metrics.get('users_mof', 0)} did MOF\n"
        )

        # b. Conversations section
        message += (
            f"b. Convos: {metrics.get('conversations', 0) + metrics.get('follow_up', 0)}, "
            f"{metrics.get('conversations', 0)} new chats, "
            f"{metrics.get('follow_up', 0)} follow ups, "
            f"{metrics.get('show_more', 0)} SM\n"
        )

        # c. Models section
        message += (
            f"c. Models: {metrics.get('models_used', 0)} used, "
            f"{metrics.get('streaming_complete', 0)} completes, "
            f"{metrics.get('streaming_errors', 0)} errors, "
            f"{metrics.get('streaming_stopped', 0)} stopped\n"
        )

        # d. QT section
        message += (
            f"d. QT: {metrics.get('qt_refusals', 0)} refusals, "
            f"{metrics.get('qt_latency', 0)} high latency, "
            f"{metrics.get('qt_evals', 0)} evals, "
            f"{metrics.get('qt_thumbs_up', 0)} thumbs up, "
            f"{metrics.get('qt_thumbs_down', 0)} thumbs down\n"
        )

        # e. Feedbacks section
        message += (
            f"e. Feedbacks: {metrics.get('feedbacks_pref', 0)} PREF, "
            f"{metrics.get('feedbacks_mof', 0)} MOFs, "
            f"{metrics.get('feedbacks_af', 0)} AF\n"
        )

        # f. Credits section
        message += (
            f"f. Credits: {metrics.get('credits_qt',0)} QT, "
            f"{metrics.get('credits_af', 0)} AF, "
            f"{metrics.get('credits_pref', 0)} PREF, "
            f"{metrics.get('credits_total', 0)} Total\n"
        )

        # g. Cashout section
        message += (
            f"g. Cashout: {metrics.get('cashout_users', 0)} users, "
            f"{metrics.get('cashout_count', 0)} cashouts, "
            f"{metrics.get('cashout_amount', 0)} credits initiated\n"
        )

        # h. Waitlist section
        message += (
            f"h. Waitlist: {metrics.get('waitlist_sign_up', 0)} new google sign-ins, "
            f"{metrics.get('waitlist_sign_up_no_google', 0)} joined waitlist without Google account\n"
        )

        # i. Invite section
        message += (
            f"i. SIC: {metrics.get('sic_submitted_valid', 0)} SICs accepted"
            f" out of {metrics.get('sic_submitted_total', 0)} submitted,"
            f" {metrics.get('sic_post_signup_chat', 0)} / {metrics.get('sic_submitted_valid', 1)} sent prompt,"
            f" {metrics.get('sic_post_signup_pref', 0)} / {metrics.get('sic_submitted_valid', 1)} did PREF\n"
        )
        logging.info(json_dumps(log_dict))
        analytics_webhook_url = os.environ.get("ANALYTICS_SLACK_WEBHOOK_URL")
        await post_to_slack(message, analytics_webhook_url)

    except Exception as e:
        error_message = f"⚠️ Failed to process chart data: {e}"
        logging.error(error_message)


async def post_data_from_cohorts(auth: str, start_date: datetime, end_date: datetime) -> None:
    """Fetch metrics for multiple Amplitude cohorts and post them to Slack."""
    message = f"*Guest report for {start_date.date()}*"
    log_dict: dict[str, int] = {}

    try:
        # Only process weekly cohorts on Sunday (weekday 6)
        cohorts_items = [
            (name, info)
            for name, info in AMPLITUDE_COHORTS.items()
            if not name.startswith("weekly_") or start_date.weekday() == 6
        ]
        metrics: dict[str, int] = {}

        for cohort_name, cohort_info in cohorts_items:
            try:
                cohort_data = fetch_cohort_data(
                    cohort_id=cohort_info["id"], auth=auth, start_date=start_date, end_date=end_date
                )
                user_ids = cohort_data.get("ids", [])
                total_users = len(user_ids)
                metrics[cohort_name] = total_users
                message += f"\n\n*{cohort_info['description']}: {total_users}*\n\n"
                user_names_dict = await fetch_user_names(user_ids)
                message += ", ".join(sorted(user_names_dict.values(), key=lambda name: name.lower()))
            except (KeyError, RequestException) as e:
                logging.error(f"Failed to process data for cohort {cohort_name}: {e}")
                metrics[cohort_name] = 0

        logging.info(json_dumps(log_dict))
        analytics_webhook_url = os.environ.get("ANALYTICS_SLACK_WEBHOOK_URL")
        await post_to_slack(message, analytics_webhook_url)
    except Exception as e:
        error_message = f"⚠️ Failed to process cohort data: {e}"
        logging.error(error_message)


async def post_credit_metrics(start_date: datetime, end_date: datetime) -> None:
    """Fetch and post credit metrics to Slack.

    Posts daily metrics every day, and additional weekly/monthly/quarterly metrics on Sundays.
    Daily metrics only pull last 2 days of data, while weekly metrics pull last 90 days.

    Args:
        start_date: Start date for the query
        end_date: End date for the query
    """
    from sqlmodel import Session, text
    from ypl.backend.db import get_engine

    daily_query = """
    WITH base_data AS (
        SELECT
            date_trunc('day', created_at AT TIME ZONE 'America/Los_Angeles')::date AS date,
            SUM(credit_delta) AS total_credits
        FROM
            rewards
        WHERE
            status = 'CLAIMED'
            AND created_at >= NOW() - INTERVAL '2 days'
        GROUP BY
            date_trunc('day', created_at AT TIME ZONE 'America/Los_Angeles')::date
    ),
    turn_data AS (
        SELECT
            date_trunc('day', created_at AT TIME ZONE 'America/Los_Angeles')::date AS date,
            COUNT(turn_id) AS turn_count
        FROM
            turns t
        WHERE
            created_at >= NOW() - INTERVAL '2 days'
            AND deleted_at IS NULL
        GROUP BY
            date_trunc('day', created_at AT TIME ZONE 'America/Los_Angeles')::date
    ),
    combined_data AS (
        SELECT
            b.date,
            b.total_credits,
            COALESCE(t.turn_count, 0) as turn_count
        FROM base_data b
        LEFT JOIN turn_data t ON b.date = t.date
    )
    SELECT
        'daily' AS period,
        date AS period_start,
        SUM(total_credits) AS total_credits,
        SUM(turn_count) AS turn_count,
        CASE
            WHEN SUM(turn_count) > 0 THEN ROUND(SUM(total_credits)::numeric / SUM(turn_count))::integer
            ELSE 0
        END AS credits_per_turn
    FROM
        combined_data
    GROUP BY
        date
    ORDER BY
        date DESC;
    """

    weekly_query = """
    WITH base_data AS (
        SELECT
            date_trunc('day', created_at AT TIME ZONE 'America/Los_Angeles')::date AS date,
            SUM(credit_delta) AS total_credits
        FROM
            rewards
        WHERE
            status = 'CLAIMED'
            AND created_at >= NOW() - INTERVAL '90 days'
        GROUP BY
            date_trunc('day', created_at AT TIME ZONE 'America/Los_Angeles')::date
    ),
    turn_data AS (
        SELECT
            date_trunc('day', created_at AT TIME ZONE 'America/Los_Angeles')::date AS date,
            COUNT(turn_id) AS turn_count
        FROM
            turns t
        WHERE
            created_at >= NOW() - INTERVAL '90 days'
            AND deleted_at IS NULL
        GROUP BY
            date_trunc('day', created_at AT TIME ZONE 'America/Los_Angeles')::date
    ),
    combined_data AS (
        SELECT
            b.date,
            b.total_credits,
            COALESCE(t.turn_count, 0) as turn_count
        FROM base_data b
        LEFT JOIN turn_data t ON b.date = t.date
    )
    SELECT
        'weekly' AS period,
        date_trunc('week', date)::date AS period_start,
        SUM(total_credits) AS total_credits,
        SUM(turn_count) AS turn_count,
        CASE
            WHEN SUM(turn_count) > 0 THEN ROUND(SUM(total_credits)::numeric / SUM(turn_count))::integer
            ELSE 0
        END AS credits_per_turn
    FROM
        combined_data
    GROUP BY
        date_trunc('week', date)
    UNION ALL
    SELECT
        'monthly' AS period,
        date_trunc('month', date)::date AS period_start,
        SUM(total_credits) AS total_credits,
        SUM(turn_count) AS turn_count,
        CASE
            WHEN SUM(turn_count) > 0 THEN ROUND(SUM(total_credits)::numeric / SUM(turn_count))::integer
            ELSE 0
        END AS credits_per_turn
    FROM
        combined_data
    GROUP BY
        date_trunc('month', date)
    UNION ALL
    SELECT
        'quarterly' AS period,
        date_trunc('quarter', date)::date AS period_start,
        SUM(total_credits) AS total_credits,
        SUM(turn_count) AS turn_count,
        CASE
            WHEN SUM(turn_count) > 0 THEN ROUND(SUM(total_credits)::numeric / SUM(turn_count))::integer
            ELSE 0
        END AS credits_per_turn
    FROM
        combined_data
    GROUP BY
        date_trunc('quarter', date)
    ORDER BY
        period_start DESC;
    """

    try:
        with Session(get_engine()) as session:
            daily_results = session.execute(text(daily_query)).all()
            daily_metrics = list(daily_results)

            if len(daily_metrics) >= 2:
                yesterday_metrics = daily_metrics[1]
                message = f"*Credit Metrics for {yesterday_metrics.period_start}*\n"
                message += (
                    f"Credits Claimed Yesterday: {yesterday_metrics.total_credits:,.0f} "
                    f"({yesterday_metrics.turn_count:,} turns, "
                    f"{yesterday_metrics.credits_per_turn:,d} credits/turn)\n"
                    f"Corresponding USD: {yesterday_metrics.total_credits * CREDITS_TO_USD_RATE:,.2f}\n"
                )

                if start_date.weekday() == 6:
                    weekly_results = session.execute(text(weekly_query)).all()
                    weekly_metrics = next((r for r in weekly_results if r.period == "weekly"), None)
                    monthly_metrics = next((r for r in weekly_results if r.period == "monthly"), None)
                    quarterly_metrics = next((r for r in weekly_results if r.period == "quarterly"), None)

                    if weekly_metrics:
                        message += (
                            f"\nWeekly Credits (Week of {weekly_metrics.period_start}): "
                            f"{weekly_metrics.total_credits:,.0f} "
                            f"({weekly_metrics.turn_count:,} turns, "
                            f"{weekly_metrics.credits_per_turn:,d} credits/turn)"
                            f"Corresponding USD: {weekly_metrics.total_credits * CREDITS_TO_USD_RATE:,.2f}\n"
                        )
                    if monthly_metrics:
                        message += (
                            f"\nMonthly Credits ({monthly_metrics.period_start.strftime('%B %Y')}): "
                            f"{monthly_metrics.total_credits:,.0f} "
                            f"({monthly_metrics.turn_count:,} turns, "
                            f"{monthly_metrics.credits_per_turn:,d} credits/turn)"
                            f"Corresponding USD: {monthly_metrics.total_credits * CREDITS_TO_USD_RATE:,.2f}\n"
                        )
                    if quarterly_metrics:
                        quarter = (quarterly_metrics.period_start.month - 1) // 3 + 1
                        message += (
                            f"\nQuarterly Credits (Q{quarter} "
                            f"{quarterly_metrics.period_start.year}): "
                            f"{quarterly_metrics.total_credits:,.0f} "
                            f"({quarterly_metrics.turn_count:,} turns, "
                            f"{quarterly_metrics.credits_per_turn:,d} credits/turn)"
                            f"Corresponding USD: {quarterly_metrics.total_credits * CREDITS_TO_USD_RATE:,.2f}\n"
                        )

                analytics_webhook_url = os.environ.get("ANALYTICS_SLACK_WEBHOOK_URL")
            else:
                message = "⚠️ No credit metrics data available"
            await post_to_slack(message, analytics_webhook_url)

    except Exception as e:
        error_message = f"⚠️ Failed to fetch credit metrics: {e}"
        logging.error(error_message)
        raise


async def post_analytics_to_slack() -> None:
    """Fetch metrics from multiple Amplitude charts and post them to Slack.

    Raises:
        ValueError: If required environment variables are not set.
        RequestException: If API requests fail.
    """
    api_key = os.environ.get("AMPLITUDE_API_KEY")
    api_secret = os.environ.get("AMPLITUDE_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError("AMPLITUDE_API_KEY and AMPLITUDE_API_SECRET must be set")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    auth = b64encode(f"{api_key}:{api_secret}".encode()).decode()

    try:
        await post_data_from_charts(auth=auth, start_date=start_date, end_date=end_date)
        await post_data_from_cohorts(auth=auth, start_date=start_date, end_date=end_date)
        await post_credit_metrics(start_date=start_date, end_date=end_date)

    except Exception as e:
        error_message = f"⚠️ Failed to fetch Amplitude metrics: {e}"
        logging.error(error_message)
        raise
