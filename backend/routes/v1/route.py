from functools import cache

from fastapi import APIRouter, Query

from backend.llm.constants import MODELS
from backend.llm.ranking import ChoixRanker
from backend.llm.routing.policy import DEFAULT_ROUTING_POLICY
from backend.llm.routing.router import RankedRouter

router = APIRouter()


@cache
def get_router() -> RankedRouter:
    ranker = ChoixRanker(
        models=MODELS,
        choix_ranker_algorithm="lsr_pairwise",
    )
    router = RankedRouter(models=MODELS, policy=DEFAULT_ROUTING_POLICY, ranker=ranker)
    # TODO(gm) replace this with actual data from the DB.
    return router


@router.post("/select_models")
def select_models(
    prompt: str = Query(..., description="Prompt"),
    num_models: int = Query(default=2, description="Number of different models to route to"),
    budget: float = Query(default=float("inf"), description="Budget"),
) -> list[str]:
    return get_router().select_models(num_models, budget=budget)


@router.post("/update_router")
def update_router(
    model_a: str = Query(..., description="Model A"),  # noqa: B008
    model_b: str = Query(..., description="Model B"),  # noqa: B008
    result: float = Query(..., description="Outcome (for Model A)"),  # noqa: B008
) -> None:
    get_router().update_ranker(model_a, model_b, result)
