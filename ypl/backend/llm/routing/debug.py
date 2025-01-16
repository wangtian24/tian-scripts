from collections import OrderedDict

from pydantic import BaseModel

from ypl.backend.llm.routing.router_state import RouterState


class ModelDebugInfo(BaseModel):
    """Debug information for a single model"""

    is_selected: bool
    is_fallback: bool
    required_by_user: bool
    journey: str
    score: float


class RoutingDebugInfo(BaseModel):
    """Stores the debug information for routing."""

    model_debug: OrderedDict[str, ModelDebugInfo]


def build_routing_debug_info(
    selected_models_rs: RouterState,
    fallback_models_rs: RouterState | None = None,
    required_models: list[str] | None = None,
) -> RoutingDebugInfo:
    """
    Builds the debug information for routing from selected and fallback RouterStates
    """
    # attach score and sort the per model debug by score
    selected_models = selected_models_rs.get_sorted_selected_models()
    fallback_models = fallback_models_rs.get_sorted_selected_models() if fallback_models_rs else []
    model_debug_list = [
        (
            model,
            ModelDebugInfo(
                is_selected=model in selected_models,
                is_fallback=model in fallback_models,
                required_by_user=required_models is not None and model in required_models,
                journey=journey_debug,
                score=selected_models_rs.model_scores[model],
            ),
        )
        for model, journey_debug in selected_models_rs.model_journey.items()
    ]
    # sort selected models by score desc first, unselected by name only
    model_debug_list.sort(key=lambda tp: (-tp[1].score, tp[0].lower()))
    return RoutingDebugInfo(model_debug=OrderedDict((k, v) for k, v in model_debug_list))
