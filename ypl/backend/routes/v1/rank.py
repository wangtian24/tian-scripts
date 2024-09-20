import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ypl.backend.llm.ranking import RatedModel, get_default_ranker, get_ranker
from ypl.db.ratings import OVERALL_CATEGORY_NAME

from ..api_auth import validate_api_key

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
router = APIRouter(dependencies=[Depends(validate_api_key)])


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

        ranker = get_default_ranker()
        ranker.add_evals_from_db(**params)
        return ranker.leaderboard()
    except Exception as e:
        logger.error(f"Error updating rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def can_use_global_rankers(params: dict[str, Any]) -> bool:
    """Returns whether the global rankers can be used for the given filters."""
    return (not params["category_names"] or params["category_names"] == [OVERALL_CATEGORY_NAME]) and not any(
        [v for k, v in params.items() if k != "category_names"]
    )
