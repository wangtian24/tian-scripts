from fastapi import APIRouter, Query

from backend.llm.ranking import RANKER_TYPES, AnnotatedFloat, RankedModel, get_ranker

router = APIRouter()


# TODO(gm): Add filters, e.g. language, model type, etc.
@router.post("/rank")
def model_rank(
    ranker_type: str = Query(default="elo", description="The ranker type", choices=RANKER_TYPES),
    model_name: str = Query(..., description="The model name"),
) -> AnnotatedFloat:
    return get_ranker(ranker_type).rank_annotate(model_name)


@router.get("/leaderboard")
def leaderboard(
    ranker_type: str = Query(default="elo", description="The ranker type", choices=RANKER_TYPES),
) -> list[RankedModel]:
    return get_ranker(ranker_type).leaderboard()
