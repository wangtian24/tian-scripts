import asyncio
import logging
import uuid
from collections.abc import Sequence
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel
from sqlmodel import select

from ypl.backend.abuse.activity import SHORT_TIME_WINDOWS, check_activity_volume_abuse
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
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
from ypl.backend.llm.routing.features import RequestContext, collect_model_features, model_has_abilities
from ypl.backend.llm.routing.reasons import summarize_reasons
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.routing.router_state import RouterState
from ypl.backend.llm.turn_quality import label_turn_quality
from ypl.backend.utils.async_utils import create_background_task
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.monitoring import metric_inc, metric_inc_by
from ypl.backend.utils.utils import StopWatch
from ypl.db.chats import (
    PromptModifier,
)
from ypl.db.routing_info import RoutingInfo

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


class RoutingPayload(BaseModel):
    prompt_categories: list[str]


class SelectModelsV2Response(BaseModel):
    # legacy model information
    models: list[tuple[str, list[tuple[str, str]]]]  # list of (model, list[(prompt modifier ID, prompt modifier)])
    provider_map: dict[str, str]  # map from model to provider
    fallback_models: list[tuple[str, list[tuple[str, str]]]]  # list of fallback models and modifiers

    # new version of model information, added on 2/1/25, we shall migrate to this gradually
    selected_models: list[SelectedModelInfo]

    routing_debug_info: RoutingDebugInfo | None = None
    num_models_remaining: int | None = None
    routing_info_id: uuid.UUID | None = None

    # An opaque payload with routing data, the client shall take this and pass back as is at the chat_completion time.
    routing_payload: dict[str, Any] | None = None


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
        create_background_task(store_modifiers(request.turn_id, prompt_modifiers))

    return prompt_modifiers


def kick_off_label_turn_quality(prompt: str, chat_id: str, turn_id: str) -> str:
    assert prompt is not None, "prompt is required for NEW_CHAT or NEW_TURN intent"
    create_background_task(label_turn_quality(UUID(turn_id), UUID(chat_id), prompt, sleep_seconds=0.5))
    return prompt


def create_selector_from_request(request: SelectModelsV2Request) -> list[ModelAndStyleSelector]:
    """
    Create a list of selector from the old fields in the request (required_models and prompt_modifier_id),
    this is just for the backward compatibility so we always store the routing info in the DB in the new format.
    """
    if request.required_models is None and request.prompt_modifier_id is None:
        return []

    return (
        [ModelAndStyleSelector(model=m) for m in request.required_models]
        if request.required_models
        else [ModelAndStyleSelector(modifier_id=request.prompt_modifier_id)]
    )


async def store_routing_info(
    routing_info_id: uuid.UUID,
    turn_id: str,
    selectors: list[ModelAndStyleSelector],
    all_categories: list[str],
    selected_model_infos: list[SelectedModelInfo],
) -> None:
    """
    Write routing info to the database
    """
    try:
        routing_info = RoutingInfo(
            routing_info_id=routing_info_id,  # we generate this ID manually
            turn_id=UUID(turn_id),
            selector=[s.model_dump(exclude_unset=True, mode="json") for s in selectors],
            categories=all_categories or [],
            routing_outcome=[m.model_dump(exclude_unset=True, mode="json") for m in selected_model_infos],
        )
        async with get_async_session() as session:
            session.add(routing_info)
            await session.commit()
    except Exception as e:
        logging.error(f"Error storing routing info: {e}")


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
        create_background_task(check_activity_volume_abuse(request.user_id, time_windows=SHORT_TIME_WINDOWS))

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

    # Kick off prompt labeling and modifier tasks.
    # Do the prompt labeling, and merge with the passed-in categories (like 'image' and 'pdf', as only
    # Frontend knows about them at this moment.
    all_categories = await get_prompt_categories(prompt)
    all_categories = list(dict.fromkeys((all_categories or []) + (request.provided_categories or [])))
    prompt_modifiers_task = asyncio.create_task(get_prompt_modifiers(prompt))

    def _remove_same_turn_shown_models(models: list[str]) -> list[str]:
        return [m for m in models if m not in (preference.same_turn_shown_models or [])]

    # Find out user selected models (from picker) and inherited models (inferred from past turns based on product logic)
    # See detailed logic in https://docs.google.com/document/d/1C941VwVwFFrv1k2iPMLkLN9ckaEjOlJpwfzOLwB7SsY/edit
    user_selected_models: list[str] = []
    if request.selectors:
        user_selected_models = [s.model for s in request.selectors if s.model is not None]
    elif request.required_models:
        user_selected_models = request.required_models
    else:
        # infer from the history in DB
        user_selected_models = list(await get_chat_required_models(UUID(request.chat_id), UUID(request.turn_id)))
    user_selected_models = _remove_same_turn_shown_models(user_selected_models)

    inherited_models: list[str] = preference.get_inherited_models(
        is_show_me_more=(request.intent == SelectIntent.SHOW_ME_MORE)
    )
    inherited_models = list(dict.fromkeys(inherited_models))  # just dedupe
    inherited_models = _remove_same_turn_shown_models(inherited_models)
    stopwatch.record_split("infer_required_models")

    log_dict = {
        "message": f"Model routing: prompt classification and required models for [{prompt[:100]}]",
        "chat_id": request.chat_id,
        "turn_id": request.turn_id,
        "user_id": request.user_id,
        "preference": preference,
        "labeled_categories": ", ".join(all_categories or []),
        "provided_categories": ", ".join(request.provided_categories or []),
        "inherited_models": ", ".join(inherited_models),
        "user_selected_models": ", ".join(user_selected_models),
    }
    logging.info(json_dumps(log_dict))
    stopwatch.record_split("prompt_classification")

    # Check if our current model set already has enough abilities to process all categories
    model_features = await collect_model_features()
    required_models_for_routing = user_selected_models + inherited_models
    has_enough_abilities = all(
        model_has_abilities(model, all_categories, model_features) for model in required_models_for_routing
    )

    # Run the routing chain if we don't have enough models to serve.
    primary_models = []
    fallback_models = []
    models_rs = None

    if len(required_models_for_routing) > request.num_models * 2 and has_enough_abilities:
        # Allow routing to be short-circuited if we don't need any spacial abilities.
        primary_models = required_models_for_routing[: request.num_models]
        fallback_models = required_models_for_routing[request.num_models : request.num_models * 2]
        num_models_remaining = request.num_models  # since we didn't run the chain, we just always assume there's enough
    else:
        # create a router chain
        router = await get_simple_pro_router(
            prompt,
            request.num_models,
            preference,
            user_selected_models=user_selected_models,
            inherited_models=inherited_models,
            same_turn_shown_models=preference.same_turn_shown_models or [],
            provided_categories=all_categories,
            chat_id=request.chat_id,
            intent=request.intent,
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
    prompt_modifiers = await prompt_modifiers_task
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
    reasons_by_model = await summarize_reasons(request_context, model_features, models_rs)

    # Prepare the response
    selected_model_infos = [
        SelectedModelInfo(
            model=model,
            provider=providers_by_model[model],
            type=SelectedModelType.PRIMARY if model in primary_models else SelectedModelType.FALLBACK,
            prompt_style_modifier_id=_get_modifier_id(model),
            reasons=reasons_by_model[model].reasons or [],
            reason_desc=reasons_by_model[model].description or "",
        )
        for model in primary_models + fallback_models
    ]

    # We generate this ID first so as to return from routing earlier so we don't block on it.
    # The client will take this ID and send it to the chat_completion endpoint later.
    routing_info_id: UUID = uuid.uuid4()

    response = SelectModelsV2Response(
        models=[(model, prompt_modifiers_by_model.get(model, [])) for model in primary_models],
        fallback_models=[(model, prompt_modifiers_by_model.get(model, [])) for model in fallback_models],
        provider_map=providers_by_model,
        selected_models=selected_model_infos,
        routing_debug_info=routing_debug_info if request.debug_level > 0 else None,
        num_models_remaining=num_models_remaining,
        routing_info_id=routing_info_id,
        routing_payload=RoutingPayload(prompt_categories=all_categories).model_dump(exclude_unset=True, mode="json"),
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

    # Kick off the storage of the routing info
    # TODO(Tian): here we convert all required models to the selector format for backward compatibility,
    # this can be removed once we fully migrated to the new format in the new prompt box.
    selectors = request.selectors or create_selector_from_request(request)
    create_background_task(
        store_routing_info(routing_info_id, request.turn_id, selectors, all_categories, selected_model_infos)
    )

    return response


# TODO(bhanu) - add retry logic -
async def get_active_prompt_modifiers() -> Sequence[PromptModifier]:
    async with get_async_session() as session:
        result = await session.exec(select(PromptModifier).where(PromptModifier.deleted_at.is_(None)))  # type: ignore
        return result.all()
