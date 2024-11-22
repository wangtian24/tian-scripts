from typing import Any

from fastapi import APIRouter, Body, Query
from pydantic import BaseModel
from tqdm import tqdm

from ypl.backend.config import settings
from ypl.backend.llm.ranking import get_ranker
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.routing.router import RouterState, get_prompt_conditional_router, get_router_ranker

router = APIRouter()


class SelectModelsAndPromptsResponse(BaseModel):
    models_and_prompts: list[tuple[str, str]]


@router.post("/select_models")
def select_models(
    prompt: str = Query(..., description="Prompt"),
    num_models: int = Query(default=2, description="Number of different models to route to"),
    budget: float = Query(default=float("inf"), description="Budget"),
    preference: None | RoutingPreference = Body(default=None, description="List of past outcomes"),  # noqa: B008
) -> list[str]:
    if settings.ROUTING_USE_PROMPT_CONDITIONAL:
        router = get_prompt_conditional_router(prompt, num_models, preference)
    else:
        router, ranker = get_router_ranker()

    all_models_state = RouterState.new_all_models_state()
    selected_models = router.select_models(state=all_models_state)
    return_models = selected_models.get_sorted_selected_models()
    return return_models


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
