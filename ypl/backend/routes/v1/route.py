from typing import Any

from fastapi import APIRouter, Body, Query
from tqdm import tqdm

from ypl.backend.llm.chat import SelectModelsV2Request, SelectModelsV2Response, select_models_plus
from ypl.backend.llm.ranking import get_ranker
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.routing.router import (
    get_simple_pro_router,
)
from ypl.backend.llm.routing.router_state import RouterState

router = APIRouter()


@router.post("/select_models")
async def select_models(
    prompt: str = Query(..., description="Prompt"),
    num_models: int = Query(default=2, description="Number of different models to route to"),
    preference: None | RoutingPreference = Body(default=None, description="List of past outcomes"),  # noqa: B008
) -> list[str]:
    router = await get_simple_pro_router(
        prompt,
        num_models,
        preference or RoutingPreference(turns=[], user_id=None, same_turn_shown_models=[]),
    )
    all_models_state = await RouterState.new_all_models_state()
    selected_models = router.select_models(state=all_models_state)
    return_models = selected_models.get_sorted_selected_models()

    return return_models


@router.post("/select_models_plus")
async def select_models_plus_route(request: SelectModelsV2Request) -> SelectModelsV2Response:
    return await select_models_plus(request)


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
