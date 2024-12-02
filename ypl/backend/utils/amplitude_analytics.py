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


AMPLITUDE_CHARTS: Final[dict[str, ChartInfo]] = {
    "users": {"id": "t2bandy1", "description": "users"},
    "conversations": {"id": "ifpv0bg1", "description": "conversations"},
    "prefs": {"id": "ht7fo2h7", "description": "prefs"},
    "scratchcards": {"id": "yxihkyvd", "description": "scratchcards"},
    "credits": {"id": "hqa0es1y", "description": "credits given"},
    "feedbacks": {"id": "ezl4kq8l", "description": "app feedbacks"},
    "turns": {"id": "cmqeokju", "description": "turns"},
    "show_more": {"id": "wycwc81n", "description": "show more"},
    "unique_number_of_models": {"id": "czeomci2", "description": "unique number of models"},
    "unique_number_of_model_responses": {"id": "x5h3txne", "description": "unique number of model responses"},
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

        for i, (metric, chart_info) in enumerate(charts_items):
            try:
                data = fetch_chart_data(chart_id=chart_info["id"], auth=auth, start_date=start_date, end_date=end_date)

                yesterdays_value = int(data["data"]["series"][0][-2]["value"])
                separator = ", " if i < len(charts_items) - 1 else ""
                message += f"{yesterdays_value} {chart_info['description']}{separator}"
                log_dict[chart_info["description"]] = yesterdays_value

            except (KeyError, IndexError) as e:
                logging.error(f"Failed to process data for {metric}: {e}")
                message += f"• {chart_info['description']}: Failed to fetch, "
            except RequestException as e:
                logging.error(f"Failed to fetch {metric} data: {e}")
                message += f"• {chart_info['description']}: Failed to fetch, "

        logging.info(json_dumps(log_dict))
        analytics_webhook_url = os.environ.get("ANALYTICS_SLACK_WEBHOOK_URL")
        await post_to_slack(message, analytics_webhook_url)

    except Exception as e:
        error_message = f"⚠️ Failed to fetch Amplitude metrics: {e}"
        logging.error(error_message)
        raise
