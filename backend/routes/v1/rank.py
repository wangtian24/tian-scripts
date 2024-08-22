from fastapi import APIRouter, Query

from backend.llm.ranking import RANKER_TYPES, AnnotatedFloat, RatedModel, get_per_category_rankers

router = APIRouter()


# TODO(gm): Add filters, e.g. language, model type, etc.
@router.post("/rank")
def model_rank(
    ranker_type: str = Query(default="elo", description="The ranker type", choices=RANKER_TYPES),
    model_name: str = Query(..., description="The model name"),
) -> dict[str, AnnotatedFloat]:
    return get_per_category_rankers(ranker_type).get_annotated_rating_all_categories(model_name)


@router.get("/leaderboard")
def leaderboard(
    ranker_type: str = Query(default="elo", description="The ranker type", choices=RANKER_TYPES),
) -> dict[str, list[RatedModel]]:
    return get_per_category_rankers(ranker_type).leaderboard_all_categories()
