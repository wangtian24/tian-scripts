from functools import cache

from fastapi import APIRouter, Query
from mabwiser.mab import LearningPolicy

from backend.llm.constants import MODELS
from backend.llm.mab_router import MABRouter

router = APIRouter()


@cache
def get_mab_router() -> MABRouter:
    mab_router = MABRouter(
        arms=MODELS,
        learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.2),
    )
    mab_router.fit(MODELS, [1.0] * len(MODELS))
    return mab_router


@router.post("/select_models")
def select_models(
    prompt: str = Query(..., description="Prompt"),
    num_arms: int = Query(default=2, description="Number of different arms to route to"),
    budget: float = Query(default=float("inf"), description="Budget"),
) -> list[str]:
    return get_mab_router().select_arms(num_arms, budget=budget)


@router.post("/update_router")
def update_router(
    decisions: list[str] = Query(..., description="Selected models"),  # noqa: B008
    rewards: list[float] = Query(..., description="Rewards for the selected models"),  # noqa: B008
) -> None:
    get_mab_router().update(decisions, rewards)
    # TODO(gm): Update the ranker.
