from fastapi import APIRouter, Query

from backend.llm.ranking import AnnotatedFloat, RatedModel, get_ranker

router = APIRouter()


# TODO(gm): Add filters, e.g. language, model type, etc.
@router.post("/rank")
def model_rank(
    model_name: str = Query(..., description="The model name"),
) -> dict[str, AnnotatedFloat]:
    return get_ranker().get_annotated_rating_all_categories(model_name)


@router.get("/leaderboard")
def leaderboard(
    category: str | None = Query(default=None, description="The category to filter by"),
) -> dict[str, list[RatedModel]] | list[RatedModel]:
    if category is None:
        return get_ranker().leaderboard_all_categories()
    return get_ranker().leaderboard_category(category)
