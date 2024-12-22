import logging
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ypl.backend.llm.ranking import RatedModel, get_default_ranker, get_ranker
from ypl.backend.utils.json import json_dumps
from ypl.db.ratings import OVERALL_CATEGORY_NAME

router = APIRouter()


@router.get("/instarank")
async def leaderboard(
    category_names: list[str] | None = Query(default=None, description="The categories to filter by"),  # noqa: B008
    exclude_ties: bool = Query(default=False, description="Whether to exclude ties"),  # noqa: B008
    from_date: datetime | None = Query(default=None, description="The prompt start date to filter by"),  # noqa: B008
    to_date: datetime | None = Query(default=None, description="The prompt end date to filter by"),  # noqa: B008
    user_from_date: datetime | None = Query(default=None, description="The user start date to filter by"),  # noqa: B008
    user_to_date: datetime | None = Query(default=None, description="The user end date to filter by"),  # noqa: B008
    language_codes: list[str] | None = Query(default=None, description="The language codes to filter by"),  # noqa: B008
) -> dict[str, list[RatedModel]] | list[RatedModel]:
    try:
        params = locals()
        if can_use_global_rankers(params):
            return get_ranker().leaderboard_all_categories()

        start_time = time.time()
        ranker = get_default_ranker()
        ranker.add_evals_from_db(**params)
        end_time = time.time()
        latency = round(end_time - start_time, 3)
        log_dict = {
            "message": f"Instarank leaderboard latency: {latency} seconds",
            "instarank_latency": latency,
        }
        logging.info(json_dumps(log_dict))
        return ranker.leaderboard()
    except Exception as e:
        log_dict = {
            "message": f"Error updating rankings: {e}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


def can_use_global_rankers(params: dict[str, Any]) -> bool:
    """Returns whether the global rankers can be used for the given filters."""
    return (not params["category_names"] or params["category_names"] == [OVERALL_CATEGORY_NAME]) and not any(
        [v for k, v in params.items() if k != "category_names"]
    )
