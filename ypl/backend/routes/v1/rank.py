import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ypl.backend.llm.ranking import RatedModel, get_default_ranker, get_ranker
from ypl.db.ratings import OVERALL_CATEGORY_NAME

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
router = APIRouter()


class LeaderboardRequest(BaseModel):
    category_names: list[str] | None
    exclude_ties: bool = False
    language: str | None = None
    model_names: list[str] | None = None
    from_date: datetime | None = None
    to_date: datetime | None = None
    user_from_date: datetime | None = None
    user_to_date: datetime | None = None


def can_use_global_rankers(request: LeaderboardRequest) -> bool:
    """Returns whether the global rankers can be used for the given filters."""
    return (not request.category_names or request.category_names == [OVERALL_CATEGORY_NAME]) and not any(
        [
            request.from_date,
            request.to_date,
            request.user_from_date,
            request.user_to_date,
            request.exclude_ties,
            request.language,
            request.model_names,
        ]
    )


@router.post("/leaderboard")
async def leaderboard(request: LeaderboardRequest) -> dict[str, list[RatedModel]] | list[RatedModel]:
    try:
        if can_use_global_rankers(request):
            return get_ranker().leaderboard_all_categories()
        ranker = get_default_ranker()
        ranker.add_evals_from_db(
            request.category_names,
            request.exclude_ties,
            request.language,
            request.model_names,
            request.from_date,
            request.to_date,
            request.user_from_date,
            request.user_to_date,
        )
        return ranker.leaderboard()
    except Exception as e:
        logger.error(f"Error updating rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
