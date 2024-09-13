import logging

from fastapi import APIRouter, HTTPException, Query

from backend.llm.ranking import RatedModel, can_use_global_rankers, get_default_ranker, get_ranker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
router = APIRouter()


@router.post("/leaderboard")
def leaderboard(
    category_names: list[str] | None = Query(default=None, description="The categories to filter by"),  # noqa: B008
    exclude_ties: bool = Query(default=False, description="Whether to exclude ties"),
    language: str | None = Query(default=None, description="The language to filter by"),
    model_names: list[str] | None = Query(default=None, description="The models to filter by"),  # noqa: B008
) -> dict[str, list[RatedModel]] | list[RatedModel]:
    try:
        if can_use_global_rankers(category_names, exclude_ties, language, model_names):
            return get_ranker().leaderboard_all_categories()
        ranker = get_default_ranker()
        ranker.add_evals_from_db(category_names, exclude_ties, language, model_names)
        return ranker.leaderboard()
    except Exception as e:
        logger.error(f"Error updating rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
