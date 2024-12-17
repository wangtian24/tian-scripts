import logging
import os
from base64 import b64encode
from datetime import datetime, timedelta
from typing import Any, Final

import requests
from requests.exceptions import RequestException
from typing_extensions import TypedDict
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps


class ChartInfo(TypedDict):
    id: str
    description: str
    series_number: int


AMPLITUDE_CHARTS: Final[dict[str, ChartInfo]] = {
    "users": {"id": "ruptpxku", "description": "active users", "series_number": 0},
    "users_start_chat": {"id": "ruptpxku", "description": "users start chat", "series_number": 1},
    "users_pref": {"id": "ruptpxku", "description": "users pref", "series_number": 2},
    "users_mof": {"id": "ruptpxku", "description": "users mof", "series_number": 3},
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
}


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


async def post_amplitude_metrics_to_slack() -> None:
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
    message = f"*Amplitude Metrics for {start_date.date()}*\n"
    log_dict: dict[str, int] = {}

    try:
        charts_items = list(AMPLITUDE_CHARTS.items())
        metrics: dict[str, int] = {}

        # First collect all metrics
        for metric, chart_info in charts_items:
            try:
                data = fetch_chart_data(chart_id=chart_info["id"], auth=auth, start_date=start_date, end_date=end_date)
                series_data = data["data"]["series"][chart_info["series_number"]]
                yesterdays_value = int(series_data[-2]["value"])
                metrics[metric] = yesterdays_value
                log_dict[chart_info["description"]] = yesterdays_value
            except (KeyError, IndexError, RequestException) as e:
                logging.error(f"Failed to process data for {metric}: {e}")
                metrics[metric] = 0

        # Format message in sections
        message = f"*Amplitude Metrics for {start_date.date()}*\n"

        # a. Users section
        message += (
            f"a. Users: {metrics.get('users', 0)} active, "
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

        logging.info(json_dumps(log_dict))
        analytics_webhook_url = os.environ.get("ANALYTICS_SLACK_WEBHOOK_URL")
        await post_to_slack(message, analytics_webhook_url)

    except Exception as e:
        error_message = f"⚠️ Failed to fetch Amplitude metrics: {e}"
        logging.error(error_message)
        raise
