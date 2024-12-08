import logging
from enum import Enum
from typing import Any

from fastapi import APIRouter, Body, Query
from pydantic import BaseModel
from tqdm import tqdm

from ypl.backend.llm.chat import deduce_original_providers, get_preferences, get_shown_models, get_user_message
from ypl.backend.llm.prompt_selector import CategorizedPromptModifierSelector, get_modifiers_by_model, store_modifiers
from ypl.backend.llm.ranking import get_ranker
from ypl.backend.llm.routing.route_data_type import PreferredModel, RoutingPreference
from ypl.backend.llm.routing.router import (
    RouterState,
    get_simple_pro_router,
)
from ypl.backend.llm.utils import GlobalThreadPoolExecutor

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
    chat_id: str | None = None  # chat ID to use for routing
    turn_id: str | None = None  # turn ID to use for routing


class SelectModelsV2Response(BaseModel):
    models: list[tuple[str, list[tuple[str, str]]]]  # list of (model, list[(prompt modifier ID, prompt modifier)])
    provider_map: dict[str, str]  # map from model to provider


@router.post("/select_models")
def select_models(
    prompt: str = Query(..., description="Prompt"),
    num_models: int = Query(default=2, description="Number of different models to route to"),
    budget: float = Query(default=float("inf"), description="Budget"),
    preference: None | RoutingPreference = Body(default=None, description="List of past outcomes"),  # noqa: B008
) -> list[str]:
    router = get_simple_pro_router(prompt, num_models, preference)
    all_models_state = RouterState.new_all_models_state()
    selected_models = router.select_models(state=all_models_state)
    return_models = selected_models.get_sorted_selected_models()
    return return_models


@router.post("/select_models_plus")
def select_models_plus(request: SelectModelsV2Request) -> SelectModelsV2Response:
    def select_models_(
        required_models: list[str] | None = None, show_me_more_models: list[str] | None = None
    ) -> list[str]:
        num_models = request.num_models
        router = get_simple_pro_router(
            prompt,
            num_models,
            preference,
            user_selected_models=required_models,
            show_me_more_models=show_me_more_models,
        )
        all_models_state = RouterState.new_all_models_state()
        selected_models = router.select_models(state=all_models_state)
        return_models = selected_models.get_sorted_selected_models()

        return return_models

    match request.intent:
        case SelectIntent.NEW_CHAT | SelectIntent.NEW_TURN:
            assert request.prompt is not None, "prompt is required for NEW_CHAT or NEW_TURN intent"
            prompt = request.prompt
        case SelectIntent.SHOW_ME_MORE:
            assert request.turn_id is not None, "turn_id is required for SHOW_ME_MORE intent"
            prompt = get_user_message(request.turn_id)

    match request.intent:
        case SelectIntent.NEW_TURN | SelectIntent.SHOW_ME_MORE:
            preference, user_selected_models = get_preferences(request.chat_id)  # type: ignore[arg-type]
            request.required_models = list(dict.fromkeys((request.required_models or []) + user_selected_models))
        case _:
            preference = RoutingPreference(turns=[])

    show_me_more_models = []

    if request.intent == SelectIntent.SHOW_ME_MORE:
        shown_models = get_shown_models(request.turn_id)  # type: ignore[arg-type]

        if preference.turns is None:
            preference.turns = []

        show_me_more_models = shown_models[-request.num_models :]
        preference.turns.append(PreferredModel(models=list(dict.fromkeys(shown_models)), preferred=None))

    if request.intent == SelectIntent.NEW_TURN and preference.turns and not preference.turns[-1].has_evaluation:
        models = list(dict.fromkeys(preference.turns[-1].models + (request.required_models or [])))
    else:
        models = request.required_models or []

    if len(models) < request.num_models:
        models = select_models_(required_models=models, show_me_more_models=show_me_more_models)

    models = models[: request.num_models]

    try:
        selector = CategorizedPromptModifierSelector.make_default_from_db()

        if request.chat_id:
            modifier_history = get_modifiers_by_model(request.chat_id)
        else:
            modifier_history = {}
        prompt_modifiers = selector.select_modifiers(models, modifier_history)

        if request.turn_id:
            GlobalThreadPoolExecutor.get_instance().submit(store_modifiers, request.turn_id, prompt_modifiers)
    except Exception as e:
        logging.error(f"Error selecting modifiers: {e}")
        prompt_modifiers = {}

    return SelectModelsV2Response(
        models=[(model, prompt_modifiers.get(model, [])) for model in models],
        provider_map=deduce_original_providers(tuple(models)),
    )


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
