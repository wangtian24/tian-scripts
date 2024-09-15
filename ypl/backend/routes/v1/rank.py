import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ypl.backend.llm.ranking import RatedModel, can_use_global_rankers, get_default_ranker, get_ranker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
router = APIRouter()


class LeaderboardRequest(BaseModel):
    category_names: list[str] | None
    exclude_ties: bool = False
    language: str | None = None
    model_names: list[str] | None = None


@router.post("/leaderboard")
async def leaderboard(request: LeaderboardRequest) -> dict[str, list[RatedModel]] | list[RatedModel]:
    try:
        if can_use_global_rankers(request.category_names, request.exclude_ties, request.language, request.model_names):
            return get_ranker().leaderboard_all_categories()
        ranker = get_default_ranker()
        ranker.add_evals_from_db(request.category_names, request.exclude_ties, request.language, request.model_names)
        return ranker.leaderboard()
    except Exception as e:
        logger.error(f"Error updating rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
