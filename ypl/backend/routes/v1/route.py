from typing import Any

from fastapi import APIRouter, Body, Query
from pydantic import BaseModel, Field
from tqdm import tqdm

from ypl.backend.config import settings
from ypl.backend.llm.ranking import get_ranker
from ypl.backend.llm.routing.router import RouterState, get_prompt_conditional_router, get_router_ranker

router = APIRouter()


class PreferredModel(BaseModel):
    models: list[str] = Field(description="List of models presented to the user for a given turn.")
    preferred: str | None = Field(description="Which model was preferred by the user, or None if all are bad")


class RoutingPreference(BaseModel):
    turns: list[PreferredModel] | None = Field(
        description=(
            "The preference for each of the past turns in the chat context "
            "in chronological order (first turn is the oldest). "
            "An empty list indicates that there were no prior turns."
        )
    )


@router.post("/select_models")
async def select_models(
    prompt: str = Query(..., description="Prompt"),
    num_models: int = Query(default=2, description="Number of different models to route to"),
    budget: float = Query(default=float("inf"), description="Budget"),
    preference: None | RoutingPreference = Body(default=None, description="List of past outcomes"),  # noqa: B008
) -> list[str]:
    if settings.ROUTING_USE_PROMPT_CONDITIONAL:
        router = get_prompt_conditional_router(prompt, num_models)
    else:
        router, ranker = get_router_ranker()

    all_models_state = await RouterState.new_all_models_state()

    return (await router.aselect_models(state=all_models_state)).get_sorted_selected_models()


@router.post("/update_ranker")
def update_ranker(
    model_a: str = Query(..., description="Model A"),  # noqa: B008
    model_b: str = Query(..., description="Model B"),  # noqa: B008
    result: float = Query(..., description="Outcome (for Model A)"),  # noqa: B008
) -> None:
    get_ranker().update(model_a, model_b, result)


@router.post("/bulk_update_ranker")
def bulk_update_ranker(
    updates: list[dict[str, Any]] = Body(  # noqa: B008
        ..., description="List of updates, each in the input format of `update_ranker`."
    ),
) -> None:
    for update in tqdm(updates, desc="Updating ranker", total=len(updates)):
        get_ranker().update(
            model_a=update["model_a"],
            model_b=update["model_b"],
            result_a=update["result"],
        )
