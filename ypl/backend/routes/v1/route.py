from enum import Enum
from typing import Any

from fastapi import APIRouter, Body, Query
from pydantic import BaseModel
from tqdm import tqdm

from ypl.backend.config import settings
from ypl.backend.llm.chat import get_user_message
from ypl.backend.llm.prompt_selector import CategorizedPromptModifierSelector
from ypl.backend.llm.ranking import get_ranker
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.routing.router import RouterState, get_prompt_conditional_router, get_router_ranker

router = APIRouter()


class SelectIntent(str, Enum):
    NEW_CHAT = "new_chat"
    NEW_TURN = "new_turn"
    SHOW_ME_MORE = "show_me_more"


class SelectModelsV2Request(BaseModel):
    intent: SelectIntent
    prompt: str | None = None  # prompt to use for routing
    num_models: int = 2  # number of models to select
    required_models: list[str] | None = None  # models selected explicitly by the user
    turn_id: str | None = None  # turn ID to use for routing
    preference: None | RoutingPreference = None


class SelectModelsV2Response(BaseModel):
    models: list[tuple[str, str]]  # list of (model, prompt modifier)


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


@router.post("/select_models_plus")
def select_models_plus(request: SelectModelsV2Request) -> SelectModelsV2Response:
    match request.intent:
        case SelectIntent.NEW_CHAT | SelectIntent.NEW_TURN:
            assert request.prompt is not None, "prompt is required for NEW_CHAT or NEW_TURN intent"
            prompt = request.prompt
        case SelectIntent.SHOW_ME_MORE:
            assert request.turn_id is not None, "turn_id is required for SHOW_ME_MORE intent"
            prompt = get_user_message(request.turn_id)

    models = select_models(prompt, request.num_models, float("inf"), request.preference)
    selector = CategorizedPromptModifierSelector.make_default_from_db()
    model_mod_map = selector.select_modifiers(models)

    return SelectModelsV2Response(models=[(model, model_mod_map[model]) for model in models])


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
