import asyncio
import logging
import uuid
from enum import Enum
from uuid import UUID

from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.abuse.activity import SHORT_TIME_WINDOWS, check_activity_volume_abuse
from ypl.backend.config import settings
from ypl.backend.db import get_async_engine
from ypl.backend.llm.category_labeler import get_prompt_categories
from ypl.backend.llm.db_helpers import (
    deduce_original_providers,
    get_chat_required_models,
    get_preferences,
    get_user_message,
    notnull,
)
from ypl.backend.llm.prompt_modifier import get_prompt_modifiers
from ypl.backend.llm.prompt_selector import (
    CategorizedPromptModifierSelector,
    get_modifiers_by_model_and_position,
    store_modifiers,
)
from ypl.backend.llm.routing.common import SelectIntent
from ypl.backend.llm.routing.debug import RoutingDebugInfo, build_routing_debug_info
from ypl.backend.llm.routing.features import RequestContext, collect_model_features
from ypl.backend.llm.routing.reasons import summarize_reasons
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.routing.router import needs_special_ability
from ypl.backend.llm.routing.router_state import RouterState
from ypl.backend.llm.turn_quality import label_turn_quality
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.monitoring import metric_inc, metric_inc_by
from ypl.backend.utils.utils import StopWatch
from ypl.db.chats import (
    PromptModifier,
)

MAX_LOGGED_MESSAGE_LENGTH = 200

HORIZONTAL_RULE = "---"
RESPONSE_SEPARATOR = f"\n\n{HORIZONTAL_RULE}\n\n"

ALL_MODELS_IN_CHAT_HISTORY_PREAMBLE = f"""
Multiple assistants responded to the user's prompt.
These responses are listed below and separated by "{HORIZONTAL_RULE}".
"""


class ModelAndStyleSelector(BaseModel):
    # choose a combination of a model selector and a modifier style, at least one of them must be provided
    # the model selector, today this is just model internal_name, we will support other selectors syntax in the future
    model: str | None = None
    # the style modifier UUID
    modifier_id: uuid.UUID | None = None


class SelectModelsV2Request(BaseModel):
    user_id: str | None = None
    intent: SelectIntent
    prompt: str | None = None  # prompt to use for routing
    num_models: int = 2  # number of models to select
    required_models: list[str] | None = None  # models selected explicitly by the user, these are internal names!
    chat_id: str  # chat ID to use for routing
    turn_id: str  # turn ID to use for routing
    provided_categories: list[str] | None = None  # categories provided by the user
    debug_level: int = 0  # 0: return no debug info, log only, 1: return debug
    prompt_modifier_id: uuid.UUID | None = None  # prompt modifier ID to apply to all models
    # whether to set prompt modifiers automatically, if prompt_modifier_id is not provided
    auto_select_prompt_modifiers: bool = True

    # model and style selectors to use for routing
    # TODO(Tian): this will eventually replace required_models and prompt_modifier_id
    selectors: list[ModelAndStyleSelector] | None = None


class SelectedModelType(str, Enum):
    PRIMARY = "primary"
    FALLBACK = "fallback"


class SelectedModelInfo(BaseModel):
    model: str  # the model name
    provider: str  # the provider name
    type: SelectedModelType
    prompt_style_modifier_id: str | None  # the prompt style modifier
    reasons: list[str]  # components of the reasons
    reason_desc: str  # the description of the reason


class SelectModelsV2Response(BaseModel):
    # legacy model information
    models: list[tuple[str, list[tuple[str, str]]]]  # list of (model, list[(prompt modifier ID, prompt modifier)])
    provider_map: dict[str, str]  # map from model to provider
    fallback_models: list[tuple[str, list[tuple[str, str]]]]  # list of fallback models and modifiers

    # new version of model information, added on 2/1/25, we shall migrate to this gradually
    selected_models: list[SelectedModelInfo]

    routing_debug_info: RoutingDebugInfo | None = None
    num_models_remaining: int | None = None


def _get_modifier(
    modifier_id: str, modifier_selector: CategorizedPromptModifierSelector, request: SelectModelsV2Request
) -> PromptModifier | None:
    if not modifier_id:
        return None
    if modifier_id in modifier_selector.modifiers_by_id:
        return modifier_selector.modifiers_by_id[modifier_id]
    else:
        logging.warning(
            json_dumps(
                {
                    "message": f"Unknown modifier ID: {modifier_id}",
                    "chat_id": request.chat_id,
                    "turn_id": request.turn_id,
                }
            )
        )
        return None


def _get_modifiers_from_request(
    request: SelectModelsV2Request, modifier_selector: CategorizedPromptModifierSelector
) -> tuple[PromptModifier | None, dict[str, PromptModifier]]:
    """
    Extract global and per-model modifiers from the request, filter out the ones that don't exist in the DB.
    Returns a tuple of (global_modifier_id, per_model_modifiers).
    """
    global_modifier = None
    per_model_modifiers = {}
    if request.selectors:
        for selector in request.selectors:
            if selector.model is None or selector.model == "*":
                global_modifier = _get_modifier(str(selector.modifier_id), modifier_selector, request)
            else:
                modifier = _get_modifier(str(selector.modifier_id), modifier_selector, request)
                if modifier:
                    per_model_modifiers[selector.model] = modifier
    else:
        global_modifier = _get_modifier(str(request.prompt_modifier_id), modifier_selector, request)
    return global_modifier, per_model_modifiers


async def _set_prompt_modifiers(
    request: SelectModelsV2Request,
    primary_models: list[str],
    fallback_models: list[str],
    applicable_modifiers: list[str],
) -> dict[str, list[tuple[str, str]]]:
    """
    Sets prompt modifiers for the selected models.

    Logic:
    - If modifiers were applied to the last two messages, use them to set the modifier for the corresponding messages
      in terms of the position of the model in the selection (LHS or RHS).
    - If prompt_modifier_id is provided, use it to set the prompt modifier for unmodified models.
    - If a model has previously been modified in a certain way, use the same modifier for it.
    - If a model is unmodified after all checks above, and `auto_select_prompt_modifiers` is set in the request,
      select a random modifier for it.

    Returns:
        A dictionary mapping each model name to its selected modifier, as a tuple of (ID, text).
    """
    prompt_modifiers: dict[str, list[tuple[str, str]]] = {}
    modifier_selector = CategorizedPromptModifierSelector.make_default_from_db()

    # extract from request, if no valid modifer could be found by those ids,
    # the request-specified modifiers will be ignored
    global_modifier, per_model_modifiers = _get_modifiers_from_request(request, modifier_selector)

    if global_modifier:
        # global modifier is set, apply it to every model unless otherwise specified
        for model in primary_models + fallback_models:
            if model in per_model_modifiers:
                modifier = per_model_modifiers[model]
                prompt_modifiers[model] = [(str(modifier.prompt_modifier_id), modifier.text)]
            else:
                prompt_modifiers[model] = [(str(global_modifier.prompt_modifier_id), global_modifier.text)]
    else:
        # no global modifier, use rules
        try:
            if request.chat_id:
                modifier_history, modifiers_by_position = await get_modifiers_by_model_and_position(request.chat_id)
            else:
                modifier_history, modifiers_by_position = {}, (None, None)

            should_auto_select_modifiers = request.auto_select_prompt_modifiers and request.intent not in (
                SelectIntent.SHOW_ME_MORE,
                SelectIntent.SHOW_MORE_WITH_SAME_TURN,
            )

            # merge the modifiers from primary and fallback models, here we treat fallback models as if they are
            # otherwise returned as the primary models.
            prompt_modifiers = modifier_selector.select_modifiers(
                primary_models,
                modifier_history,
                applicable_modifiers,
                modifiers_by_position=modifiers_by_position,
                should_auto_select_modifiers=should_auto_select_modifiers,
            ) | modifier_selector.select_modifiers(
                fallback_models,
                modifier_history,
                applicable_modifiers,
                modifiers_by_position=modifiers_by_position,
                should_auto_select_modifiers=should_auto_select_modifiers,
            )
            # override if any has been specified by the request
            for model in per_model_modifiers:
                prompt_modifiers[model] = [
                    (str(per_model_modifiers[model].prompt_modifier_id), per_model_modifiers[model].text)
                ]

        except Exception as e:
            logging.error(f"Error selecting modifiers: {e}")

    if request.turn_id and prompt_modifiers:
        asyncio.create_task(store_modifiers(request.turn_id, prompt_modifiers))

    return prompt_modifiers


def kick_off_label_turn_quality(prompt: str, chat_id: str, turn_id: str) -> str:
    assert prompt is not None, "prompt is required for NEW_CHAT or NEW_TURN intent"
    asyncio.create_task(label_turn_quality(UUID(turn_id), UUID(chat_id), prompt))
    return prompt


async def select_models_plus(request: SelectModelsV2Request) -> SelectModelsV2Response:
    """
    The main model routing function. It labels the prompt, extracts historic chat turn information
    to decide what model we should route to.
    """
    from ypl.backend.llm.routing.router import get_simple_pro_router

    logging.debug(json_dumps({"message": "select_models_plus request"} | request.model_dump(mode="json")))
    metric_inc(f"routing/intent_{request.intent}")
    stopwatch = StopWatch(f"routing/latency/{request.intent}/", auto_export=True)
    if request.user_id is not None and settings.ENVIRONMENT != "local":
        asyncio.create_task(check_activity_volume_abuse(request.user_id, time_windows=SHORT_TIME_WINDOWS))

    # Prepare the prompt and past turn information
    prompt = None
    match request.intent:
        case SelectIntent.NEW_CHAT:
            prompt = kick_off_label_turn_quality(notnull(request.prompt), request.chat_id, request.turn_id)
            preference = RoutingPreference(
                turns=[],
                same_turn_shown_models=[],
                user_id=request.user_id,
            )
        case SelectIntent.NEW_TURN:
            prompt = kick_off_label_turn_quality(notnull(request.prompt), request.chat_id, request.turn_id)
            preference = get_preferences(request.user_id, request.chat_id, request.turn_id, request.required_models)
        case SelectIntent.SHOW_ME_MORE:
            prompt = get_user_message(request.turn_id)
            preference = get_preferences(request.user_id, request.chat_id, request.turn_id, request.required_models)
        case _:
            raise ValueError(f"Unsupported intent {request.intent} for select_models_plus()")
    preference.debug_level = request.debug_level
    stopwatch.record_split("prepare_prompt_and_preference")

    # Figure out what models are required for routing (must appear)
    # 1. get it from request, if none, from DB, if none, from preference (inferred from history messages in the chat)
    # 2. add all previous turn non-downvoted models for stability (2/4/2025, see routing decision log)
    # 3. remove all same-turn already-shown user-selected models (when it's SMM)
    required_models: list[str] = []
    if request.selectors:
        required_models = [s.model for s in request.selectors if s.model is not None]
    elif request.required_models:
        required_models = request.required_models
    else:
        # infer from the history in DB
        required_models = list(await get_chat_required_models(UUID(request.chat_id), UUID(request.turn_id)))
    # construct the required model from routing, it's the combination of required models from the request and the
    # past-turn models we must inherit and use. See model decision log:
    # https://docs.google.com/document/d/1C941VwVwFFrv1k2iPMLkLN9ckaEjOlJpwfzOLwB7SsY/edit
    required_models_for_routing = list(required_models) + preference.get_inherited_models(
        is_show_me_more=(request.intent == SelectIntent.SHOW_ME_MORE)
    )
    # exclude models already used in earlier rounds in the same turn (due to "Show More AIs")
    required_models_for_routing = (
        [m for m in required_models_for_routing if m not in (preference.same_turn_shown_models or [])]
        if request.intent == SelectIntent.SHOW_ME_MORE
        else required_models_for_routing
    )
    required_models_for_routing = list(dict.fromkeys(required_models_for_routing))  # just dedupe
    stopwatch.record_split("infer_required_models")

    # Prompt labeling, generate categories and modifiers based on the prompt.
    prompt_categories, prompt_modifiers = await asyncio.gather(
        get_prompt_categories(prompt), get_prompt_modifiers(prompt)
    )
    # merge two sources of category labels, from latest classifier runs and from the frontend (like image and pdf)
    all_categories = (prompt_categories or []) + (request.provided_categories or [])

    log_dict = {
        "message": f"Model routing: prompt classification and required models for [{prompt[:100]}]",
        "chat_id": request.chat_id,
        "turn_id": request.turn_id,
        "user_id": request.user_id,
        "preference": preference,
        "prompt_categories": ", ".join(all_categories or []),
        "provided_categories": ", ".join(request.provided_categories or []),
        "prompt_modifiers": ", ".join(prompt_modifiers) if prompt_modifiers else "None",
        "required_models_for_routing": ", ".join(required_models_for_routing),
    }
    logging.info(json_dumps(log_dict))
    stopwatch.record_split("prompt_classification")

    # Run the routing chain if we don't have enough models to serve.
    primary_models = []
    fallback_models = []
    models_rs = None

    if len(required_models_for_routing) > request.num_models * 2 and not needs_special_ability(prompt_categories):
        # Allow routing to be short-circuited if we don't need any spacial abilities.
        primary_models = required_models_for_routing[: request.num_models]
        fallback_models = required_models_for_routing[request.num_models : request.num_models * 2]
        num_models_remaining = request.num_models  # since we didn't run the chain, we just always assume there's enough
    else:
        # create a router
        router = await get_simple_pro_router(
            prompt,
            request.num_models,
            preference,
            required_models=required_models_for_routing,
            show_me_more_models=preference.same_turn_shown_models or [],
            provided_categories=all_categories,
            chat_id=request.chat_id,
            is_new_turn=(request.intent == SelectIntent.NEW_TURN),
            with_fallback=True,
        )
        stopwatch.record_split("routing_create_chain")
        # actually run the router chain
        all_models_state = await RouterState.new_all_models_state(request.user_id)
        models_rs = router.select_models(state=all_models_state)
        # extract results
        selected_models = models_rs.get_selected_models()
        primary_models = selected_models[: request.num_models]
        fallback_models = selected_models[request.num_models : request.num_models * 2]
        num_models_remaining = (
            models_rs.num_models_remaining if models_rs.num_models_remaining is not None else request.num_models
        )
    stopwatch.record_split("routing_total")

    # Attach prompt modifiers to the models we selected
    prompt_modifiers_by_model = await _set_prompt_modifiers(request, primary_models, fallback_models, prompt_modifiers)
    stopwatch.record_split("set_prompt_modifiers")

    # Deduce the provider information for models
    providers_by_model = deduce_original_providers(tuple(primary_models + fallback_models))

    # Create debugging info and logging
    routing_debug_info: RoutingDebugInfo = build_routing_debug_info(
        primary_models=primary_models,
        fallback_models=fallback_models,
        router_state=models_rs,
        required_models=request.required_models,
    )
    if logging.root.getEffectiveLevel() == logging.DEBUG:
        for model, model_debug in routing_debug_info.model_debug.items():
            is_selected = " S => " if model in primary_models else ""
            print(f"> {model:<50}{is_selected:<8}{model_debug.score:-10.1f}{model_debug.journey}")
    if len(primary_models) != request.num_models:
        log_dict = {
            "message": f"Model routing: requested {request.num_models} models, but returned {len(primary_models)}.",
            "chat_id": request.chat_id,
            "turn_id": request.turn_id,
            "intent": request.intent,
        }
        logging.info(json_dumps(log_dict))

    def _get_modifier_id(model: str) -> str | None:
        modifiers = prompt_modifiers_by_model.get(model, [])
        # just get the first one, there's only one right now.
        return modifiers[0][0] if len(modifiers) > 0 else None

    # Summarize the reasons for routing the models
    # TODO(Tian): right now we build context and collect model features from DB just for constructing reasons,
    # later these logic will be moved to the earlier part of routing and used for the routing.
    request_context = RequestContext(
        intent=request.intent,
        user_required_models=required_models_for_routing,
        inherited_models=required_models_for_routing,
        prompt_categories=all_categories,
    )
    model_features = await collect_model_features()
    reasons_by_model = await summarize_reasons(request_context, model_features, models_rs)

    # Prepare the response
    response = SelectModelsV2Response(
        models=[(model, prompt_modifiers_by_model.get(model, [])) for model in primary_models],
        fallback_models=[(model, prompt_modifiers_by_model.get(model, [])) for model in fallback_models],
        provider_map=providers_by_model,
        selected_models=[
            SelectedModelInfo(
                model=model,
                provider=providers_by_model[model],
                type=SelectedModelType.PRIMARY if model in primary_models else SelectedModelType.FALLBACK,
                prompt_style_modifier_id=_get_modifier_id(model),
                reasons=reasons_by_model[model].reasons or [],
                reason_desc=reasons_by_model[model].description or "",
            )
            for model in primary_models + fallback_models
        ],
        routing_debug_info=routing_debug_info if request.debug_level > 0 else None,
        num_models_remaining=num_models_remaining,
    )
    response_log = {
        "message": f"Model routing final response for [{request.intent}]: prompt = [{prompt[:100]}]",
        "chat_id": request.chat_id,
        "turn_id": request.turn_id,
        "primary_models": str(primary_models),
        "fallback_models": str(fallback_models),
        "num_models_remaining": str(num_models_remaining),
        "reasons": reasons_by_model,
    }
    logging.info(json_dumps(response_log))
    stopwatch.end("prepare_response")

    metric_inc_by("routing/count_models_served", len(primary_models))
    if len(primary_models) > 0:
        metric_inc(f"routing/count_first_{primary_models[0]}")
    if len(primary_models) > 1:
        metric_inc(f"routing/count_second_{primary_models[1]}")
    for model in primary_models:
        metric_inc(f"routing/count_chosen_{model}")

    return response


# TODO(bhanu) - add retry logic -
async def get_active_prompt_modifiers() -> list[PromptModifier]:
    async with AsyncSession(get_async_engine()) as session:
        result = await session.execute(select(PromptModifier).where(PromptModifier.deleted_at.is_(None)))  # type: ignore
        return result.scalars().all()  # type: ignore
