import csv
import logging
import os
import time
from base64 import b64encode
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Final

import requests
from pytz import timezone
from requests.exceptions import RequestException
from sqlmodel import Session, text
from typing_extensions import TypedDict
from ypl.backend.db import get_engine
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.routes.v1.credit import CREDITS_TO_INR_RATE, CREDITS_TO_USD_RATE, USD_TO_INR_RATE
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import fetch_user_names


class RatioTracker:
    """Class to track various ratios throughout the analytics process."""

    def __init__(self) -> None:
        self.ratios: dict[str, float] = {}

    def add_ratio(self, name: str, numerator: int, denominator: int) -> None:
        if denominator > 0:
            self.ratios[name] = numerator / denominator

    def get_ratio_message(self) -> str:
        if not self.ratios:
            return ""

        message = "\n\n*Key Ratios:*"
        for name, value in self.ratios.items():
            message += f"\n• {name}: {value:.2%}"
        return message


ratio_tracker = RatioTracker()


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

        # Add conversation ratios
        total_convos = metrics.get("conversations", 0) + metrics.get("follow_up", 0)
        new_convos = metrics.get("conversations", 0)
        ratio_tracker.add_ratio("New/Total Conversations", new_convos, total_convos)

        #  Add QT ratios
        total_qt = total_convos
        ratio_tracker.add_ratio("QT Refusals/Total QT", metrics.get("qt_refusals", 0), total_qt)
        ratio_tracker.add_ratio("QT High Latency/Total QT", metrics.get("qt_latency", 0), total_qt)
        ratio_tracker.add_ratio("QT Evals/Total QT", metrics.get("qt_evals", 0), total_qt)
        ratio_tracker.add_ratio("QT Thumbs Up/Total QT", metrics.get("qt_thumbs_up", 0), total_qt)
        ratio_tracker.add_ratio("QT Thumbs Down/Total QT", metrics.get("qt_thumbs_down", 0), total_qt)

        # Add feedback ratios
        total_feedbacks = (
            metrics.get("feedbacks_pref", 0) + metrics.get("feedbacks_mof", 0) + metrics.get("feedbacks_af", 0)
        )
        ratio_tracker.add_ratio("PREF/Total Feedbacks", metrics.get("feedbacks_pref", 0), total_feedbacks)
        ratio_tracker.add_ratio("MOF/Total Feedbacks", metrics.get("feedbacks_mof", 0), total_feedbacks)
        ratio_tracker.add_ratio("AF/Total Feedbacks", metrics.get("feedbacks_af", 0), total_feedbacks)

        # Add SIC ratios
        ratio_tracker.add_ratio(
            "SIC->Chat Conversion", metrics.get("sic_post_signup_chat", 0), metrics.get("sic_submitted_valid", 0)
        )
        ratio_tracker.add_ratio(
            "SIC->PREF Conversion", metrics.get("sic_post_signup_pref", 0), metrics.get("sic_submitted_valid", 0)
        )

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
    """Fetch metrics for multiple Amplitude cohorts and post them to Slack.
    Always pulls both daily and weekly data. Shows daily user names every day,
    but weekly user names only on Sundays."""
    message = f"*Amplitude Guest Report for {start_date.date()}*\n"
    log_dict: dict[str, int] = {}

    try:
        cohorts_items = list(AMPLITUDE_COHORTS.items())
        metrics: dict[str, int] = {}

        for cohort_name, cohort_info in cohorts_items:
            try:
                cohort_data = fetch_cohort_data(
                    cohort_id=cohort_info["id"], auth=auth, start_date=start_date, end_date=end_date
                )
                user_ids = cohort_data.get("ids", [])
                total_users = len(user_ids)
                metrics[cohort_name] = total_users
                message += f"\n\n*{cohort_info['description']}: {total_users}*"

                if cohort_name.startswith("daily_"):
                    user_names_dict = await fetch_user_names(user_ids)
                    message += f"\n{', '.join(sorted(user_names_dict.values(), key=lambda name: name.lower()))}"
                elif cohort_name.startswith("weekly_") and start_date.weekday() == 6:
                    user_names_dict = await fetch_user_names(user_ids)
                    message += f"\n{', '.join(sorted(user_names_dict.values(), key=lambda name: name.lower()))}"

            except (KeyError, RequestException) as e:
                logging.error(f"Failed to process data for cohort {cohort_name}: {e}")
                metrics[cohort_name] = 0

        # Add DAU/WAU ratio
        daily_users = metrics.get("daily_active_guests", 0)
        weekly_users = metrics.get("weekly_active_guests", 0)
        ratio_tracker.add_ratio("DAU/WAU", daily_users, weekly_users)

        message += ratio_tracker.get_ratio_message()
        logging.info(json_dumps(log_dict))
        analytics_webhook_url = os.environ.get("ANALYTICS_SLACK_WEBHOOK_URL")
        await post_to_slack(message, analytics_webhook_url)
    except Exception as e:
        error_message = f"⚠️ Failed to process cohort data: {e}"
        logging.error(error_message)


async def post_user_base_metrics(report_date: datetime) -> None:
    """Fetch and post user base metrics to Slack, including total users and referrals."""

    user_data_query = """
         SELECT
            COUNT(*) filter (where u.status = 'ACTIVE') as total_users,
            COUNT(*) filter (
                where u.status = 'ACTIVE'
                and
                    ((created_at at TIME zone 'America/Los_Angeles')::date = :report_date)
            ) as new_actives,
            COUNT(*) filter (
            where u.status = 'DEACTIVATED') as total_users_deactivated
        FROM
            users u
        WHERE
            u.deleted_at is null;
    """

    waitlist_data_query = """
        SELECT
            COUNT(*) AS total_waitlist_users,
            COUNT(*) filter (
                where ((created_at at TIME zone 'America/Los_Angeles')::date = :report_date)
            ) as new_waitlist_users
        FROM
            waitlisted_users wu
        WHERE
            wu.status = 'PENDING'
            AND wu.deleted_at IS NULL;
    """

    yesterday_referral_data_query = """
        SELECT
            u.user_id,
            u.name,
            COUNT(*) AS referred_user_count
        FROM
            special_invite_code_claim_logs siclog
        JOIN
            special_invite_codes sic ON siclog.special_invite_code_id  = sic.special_invite_code_id
        JOIN
            users u ON sic.creator_user_id = u.user_id
        WHERE
            ((siclog.created_at at TIME zone 'America/Los_Angeles')::date = :report_date)
        GROUP BY u.user_id, name
        ORDER BY referred_user_count DESC
    """

    try:
        with Session(get_engine()) as session:
            pst_date = report_date.astimezone(timezone("America/Los_Angeles")).date()
            params = {"report_date": pst_date}

            user_data_results = session.execute(text(user_data_query), params).all()
            user_data = list(user_data_results)

            waitlist_data_results = session.execute(text(waitlist_data_query), params).all()
            waitlist_data = list(waitlist_data_results)

            message = f"*User and Waitlist Metrics for {report_date.date()}*\n"
            if len(user_data) >= 1:
                yesterday_metrics = user_data[0]
                message += f"Total users granted accecss: {yesterday_metrics.total_users:,.0f}"
                message += f" ({yesterday_metrics.new_actives:,.0f} new yesterday)\n"
                message += f"Total deactivated users: {yesterday_metrics.total_users_deactivated:,.0f}\n"
            if len(waitlist_data) >= 1:
                yesterday_metrics = waitlist_data[0]
                message += f"Total users on waitlist: {yesterday_metrics.total_waitlist_users:,.0f}"
                message += f" ({yesterday_metrics.new_waitlist_users:,.0f} new yesterday)\n"

            referral_data_results = session.execute(text(yesterday_referral_data_query), params).all()
            referral_data = list(referral_data_results)

            if len(referral_data) >= 1:
                message += f"\nNew users were invited by these {len(referral_data)} existing users:\n"
                referral_names = [f"{referral.name} ({referral.referred_user_count})" for referral in referral_data]
                message += ", ".join(referral_names)
            else:
                message += "\nNo referrals yesterday"

            analytics_webhook_url = os.environ.get("ANALYTICS_SLACK_WEBHOOK_URL")
            await post_to_slack(message, analytics_webhook_url)

    except Exception as e:
        error_message = f"⚠️ Failed to fetch user base metrics: {e}"
        logging.error(error_message)
        raise


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

    #  row means Rest of World in the query below
    daily_query = """
    WITH base_data AS (
        SELECT
            date_trunc('day', r.created_at AT TIME ZONE 'America/Los_Angeles')::date AS date,
            SUM(r.credit_delta) AS total_credits,
            SUM(CASE WHEN up.country = 'IN' THEN r.credit_delta ELSE 0 END) as total_credits_from_india,
            SUM(CASE WHEN COALESCE(up.country, 'ROW') != 'IN' THEN r.credit_delta ELSE 0 END) as total_credits_from_row
        FROM
            rewards r
        LEFT JOIN
            user_profiles up ON r.user_id = up.user_id
        WHERE
            r.status = 'CLAIMED'
            AND (r.created_at AT TIME ZONE 'America/Los_Angeles')::date =
            ((current_date - INTERVAL '1 day') AT TIME ZONE 'America/Los_Angeles')::date
        GROUP BY
            date_trunc('day', r.created_at AT TIME ZONE 'America/Los_Angeles')::date
    ),
    turn_data AS (
        SELECT
            date_trunc('day', created_at AT TIME ZONE 'America/Los_Angeles')::date AS date,
            COUNT(*) AS turn_count
        FROM
            turns t
        WHERE
            (created_at AT TIME ZONE 'America/Los_Angeles')::date =
            ((current_date - INTERVAL '1 day') AT TIME ZONE 'America/Los_Angeles')::date
            AND deleted_at IS NULL
        GROUP BY
            date_trunc('day', created_at AT TIME ZONE 'America/Los_Angeles')::date
    ),
    combined_data AS (
        SELECT
            b.date,
            b.total_credits,
            b.total_credits_from_india,
            b.total_credits_from_row,
            COALESCE(t.turn_count, 0) as turn_count
        FROM base_data b
        LEFT JOIN turn_data t ON b.date = t.date
    )
    SELECT
        'daily' AS period,
        date AS period_start,
        SUM(total_credits) AS total_credits,
        SUM(total_credits_from_india) AS total_credits_from_india,
        SUM(total_credits_from_row) AS total_credits_from_row,
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
            date_trunc('day', r.created_at AT TIME ZONE 'America/Los_Angeles')::date AS date,
            SUM(r.credit_delta) AS total_credits,
            SUM(CASE WHEN up.country = 'IN' THEN r.credit_delta ELSE 0 END) as total_credits_from_india,
            SUM(CASE WHEN COALESCE(up.country, 'ROW') != 'IN' THEN r.credit_delta ELSE 0 END) as total_credits_from_row
        FROM
            rewards r
        LEFT JOIN
            user_profiles up ON r.user_id = up.user_id
        WHERE
            r.status = 'CLAIMED'
            AND (r.created_at AT TIME ZONE 'America/Los_Angeles')::date >=
            ((current_date - INTERVAL '90 days') AT TIME ZONE 'America/Los_Angeles')::date
        GROUP BY
            date_trunc('day', r.created_at AT TIME ZONE 'America/Los_Angeles')::date
    ),
    turn_data AS (
        SELECT
            date_trunc('day', created_at AT TIME ZONE 'America/Los_Angeles')::date AS date,
            COUNT(turn_id) AS turn_count
        FROM
            turns t
        WHERE
            (created_at AT TIME ZONE 'America/Los_Angeles')::date >=
            ((current_date - INTERVAL '90 days') AT TIME ZONE 'America/Los_Angeles')::date
            AND deleted_at IS NULL
        GROUP BY
            date_trunc('day', created_at AT TIME ZONE 'America/Los_Angeles')::date
    ),
    combined_data AS (
        SELECT
            b.date,
            b.total_credits,
            b.total_credits_from_india,
            b.total_credits_from_row,
            COALESCE(t.turn_count, 0) as turn_count
        FROM base_data b
        LEFT JOIN turn_data t ON b.date = t.date
    )
    SELECT
        'weekly' AS period,
        date_trunc('week', date)::date AS period_start,
        SUM(total_credits) AS total_credits,
        SUM(total_credits_from_india) AS total_credits_from_india,
        SUM(total_credits_from_row) AS total_credits_from_row,
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
        SUM(total_credits_from_india) AS total_credits_from_india,
        SUM(total_credits_from_row) AS total_credits_from_row,
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
        SUM(total_credits_from_india) AS total_credits_from_india,
        SUM(total_credits_from_row) AS total_credits_from_row,
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

            if len(daily_metrics) >= 1:
                yesterday_metrics = daily_metrics[0]
                message = f"*Database Credit Metrics for {yesterday_metrics.period_start}*\n"
                usd_amount_india = yesterday_metrics.total_credits_from_india * CREDITS_TO_INR_RATE / USD_TO_INR_RATE
                usd_amount_row = yesterday_metrics.total_credits_from_row * CREDITS_TO_USD_RATE
                usd_amount = usd_amount_india + usd_amount_row
                message += (
                    f"Credits Claimed Yesterday: {yesterday_metrics.total_credits:,.0f} "
                    f"({yesterday_metrics.turn_count:,} turns, "
                    f"{yesterday_metrics.credits_per_turn:,d} credits/turn)\n"
                    f"Corresponding USD: {usd_amount:,.2f}\n"
                    f"(India: {usd_amount_india:,.2f}, Row: {usd_amount_row:,.2f})\n"
                )

                if start_date.weekday() == 6:
                    weekly_results = session.execute(text(weekly_query)).all()
                    weekly_metrics = next((r for r in weekly_results if r.period == "weekly"), None)
                    monthly_metrics = next((r for r in weekly_results if r.period == "monthly"), None)
                    quarterly_metrics = next((r for r in weekly_results if r.period == "quarterly"), None)

                    if weekly_metrics:
                        usd_amount_india = (
                            weekly_metrics.total_credits_from_india * CREDITS_TO_INR_RATE / USD_TO_INR_RATE
                        )
                        usd_amount_row = weekly_metrics.total_credits_from_row * CREDITS_TO_USD_RATE
                        usd_amount = usd_amount_india + usd_amount_row
                        message += (
                            f"\nWeekly Credits (Week of {weekly_metrics.period_start}): "
                            f"{weekly_metrics.total_credits:,.0f} "
                            f"({weekly_metrics.turn_count:,} turns, "
                            f"{weekly_metrics.credits_per_turn:,d} credits/turn)\n"
                            f"Corresponding USD: {usd_amount:,.2f}\n"
                            f"(India: {usd_amount_india:,.2f}, Rest of World: {usd_amount_row:,.2f})\n"
                        )
                    if monthly_metrics:
                        usd_amount_india = (
                            monthly_metrics.total_credits_from_india * CREDITS_TO_INR_RATE / USD_TO_INR_RATE
                        )
                        usd_amount_row = monthly_metrics.total_credits_from_row * CREDITS_TO_USD_RATE
                        usd_amount = usd_amount_india + usd_amount_row
                        message += (
                            f"\nMonthly Credits ({monthly_metrics.period_start.strftime('%B %Y')}): "
                            f"{monthly_metrics.total_credits:,.0f} "
                            f"({monthly_metrics.turn_count:,} turns, "
                            f"{monthly_metrics.credits_per_turn:,d} credits/turn)\n"
                            f"Corresponding USD: {usd_amount:,.2f}\n"
                            f"(India: {usd_amount_india:,.2f}, Row: {usd_amount_row:,.2f})\n"
                        )
                    if quarterly_metrics:
                        quarter = (quarterly_metrics.period_start.month - 1) // 3 + 1
                        usd_amount_india = (
                            quarterly_metrics.total_credits_from_india * CREDITS_TO_INR_RATE / USD_TO_INR_RATE
                        )
                        usd_amount_row = quarterly_metrics.total_credits_from_row * CREDITS_TO_USD_RATE
                        usd_amount = usd_amount_india + usd_amount_row
                        message += (
                            f"\nQuarterly Credits (Q{quarter} {quarterly_metrics.period_start.year}): "
                            f"{quarterly_metrics.total_credits:,.0f} "
                            f"({quarterly_metrics.turn_count:,} turns, "
                            f"{quarterly_metrics.credits_per_turn:,d} credits/turn)\n"
                            f"Corresponding USD: {usd_amount:,.2f}\n"
                            f"(India: {usd_amount_india:,.2f}, Row: {usd_amount_row:,.2f})\n"
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
        await post_user_base_metrics(report_date=start_date)
    except Exception as e:
        error_message = f"⚠️ Failed to fetch Amplitude metrics: {e}"
        logging.error(error_message)
        raise
