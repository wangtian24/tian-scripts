from functools import cache

from fastapi import APIRouter, Query
from mabwiser.mab import LearningPolicy

from backend.llm.constants import MODELS
from backend.llm.mab_ranker import MultiArmedBanditRanker
from backend.llm.routing import RankedRouter, RoutingPolicy

router = APIRouter()


@cache
def get_router() -> RankedRouter:
    ranker = MultiArmedBanditRanker(
        models=MODELS,
        learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.2),
    )
    router = RankedRouter(models=MODELS, ranker=ranker)
    battles = [(m, m) for m in MODELS]
    ranker.fit(battles, [1.0] * len(MODELS))
    return router


@router.post("/select_models")
def select_models(
    prompt: str = Query(..., description="Prompt"),
    num_models: int = Query(default=2, description="Number of different models to route to"),
    budget: float = Query(default=float("inf"), description="Budget"),
) -> list[str]:
    return get_router().select_models(num_models, budget=budget, policy=RoutingPolicy.TOP)


@router.post("/update_router")
def update_router(
    model_a: str = Query(..., description="Model A"),  # noqa: B008
    model_b: str = Query(..., description="Model B"),  # noqa: B008
    result: float = Query(..., description="Outcome (for Model A)"),  # noqa: B008
) -> None:
    get_router().ranker.update(model_a, model_b, result)
