from enum import Enum

from pydantic import BaseModel

from ypl.backend.llm.constants import IMAGE_CATEGORY, ONLINE_CATEGORY, PDF_CATEGORY
from ypl.backend.llm.routing.common import SelectIntent
from ypl.backend.llm.routing.features import ModelSetFeatures, RequestContext
from ypl.backend.llm.routing.policy import SelectionCriteria
from ypl.backend.llm.routing.router_state import RouterState


class RoutingReason(Enum):
    USER_SELECTED = "user_selected"
    PRO_OR_STRONG = "pro_or_strong"
    ONLINE_ACCESS = "online_access"
    PROCESS_IMAGE = "image"
    PROCESS_PDF = "pdf"
    CODING = "coding"
    REASONING = "reasoning"
    IMAGE_GENERATION = "image_generation"
    FAST = "fast"
    # the value should match the yapp names
    YAPP_WEATHER = "yapp-weather"
    YAPP_NEWS = "yapp-news"
    YAPP_YOUTUBE = "yapp-youtube-transcript"
    YAPP_WIKIPEDIA = "yapp-wikipedia"


# TODO(Tian): move these reasons to database once we are more settled on the structure
# Although we may return multiple reasons, but the description will only be picked from one of the reasons.
# We can consider some combinations in the future.
REASON_DESCRIPTION_TEMPLATE = {
    RoutingReason.USER_SELECTED: "You selected this model",
    RoutingReason.PRO_OR_STRONG: "This is a strong general-purpose model",
    RoutingReason.ONLINE_ACCESS: "This model can access the Internet to answer your questions",
    RoutingReason.PROCESS_IMAGE: "This model can understand images",
    RoutingReason.PROCESS_PDF: "This model can read and answer questions about PDF documents",
    RoutingReason.CODING: "This model is good at solving coding problems",
    RoutingReason.REASONING: "This model can do deep thinking and reasoning",
    RoutingReason.IMAGE_GENERATION: "This model can generate images",
    RoutingReason.FAST: "This model is blazingly fast",
    RoutingReason.YAPP_WEATHER: "This model can access latest weather information",
    RoutingReason.YAPP_NEWS: "This model can access and summarize latest news",
    RoutingReason.YAPP_YOUTUBE: "This model can transcribe and answer questions about YouTube videos",
    RoutingReason.YAPP_WIKIPEDIA: "This model is good at utilizing Wikipedia",
}

# mapping from (some) selection criteria to routing reasons. right now not all reasons are easily mappable
# from SelectionCriteria, like ROUTING_RULES can be for coding/online access and others.
# This is ordered by the priority in which we want to show them.
SELECTION_CRITERIA_TO_REASON = {
    SelectionCriteria.IMAGE_GENERATION_MODELS: RoutingReason.IMAGE_GENERATION,
    SelectionCriteria.REASONING_MODELS: RoutingReason.REASONING,
    SelectionCriteria.LIVE_MODELS: RoutingReason.ONLINE_ACCESS,
    SelectionCriteria.PRO_MODELS: RoutingReason.PRO_OR_STRONG,
    SelectionCriteria.STRONG_MODELS: RoutingReason.PRO_OR_STRONG,
    SelectionCriteria.PRO_AND_STRONG_MODELS: RoutingReason.PRO_OR_STRONG,
    SelectionCriteria.FAST_MODELS: RoutingReason.FAST,
}


class ReasonSummary(BaseModel):
    reasons: list[str]
    description: str | None


def summarize_reasons(
    request_context: RequestContext,
    model_set_features: ModelSetFeatures,
    router_state: RouterState | None = None,
) -> dict[str, ReasonSummary]:
    """
    Summarize the reasons for routing the models in the RouterState.
    Returns a dictionary of model internal_names to their reason summaries.
    """
    if not router_state:
        return {}

    result = {}

    # Go through all selected models and find out their reasons.
    for model_internal_name, criteria_map in router_state.get_selected_models_with_criteria().items():
        reasons: list[RoutingReason] = []
        features = model_set_features.model_features[model_internal_name]

        if (
            model_internal_name in request_context.user_required_models
            and request_context.intent == SelectIntent.NEW_CHAT
        ):
            # user selected model, will not consider other reasons
            reasons.append(RoutingReason.USER_SELECTED)
        elif model_internal_name not in request_context.inherited_models:
            # only generate reasons for new turns
            if "yapp-" in model_internal_name:
                # match against yapps
                for reason in RoutingReason:
                    if reason.value == model_internal_name:
                        reasons.append(reason)
                        break
            else:
                # match against model capabilities
                if IMAGE_CATEGORY in request_context.prompt_categories and features.can_process_image:
                    reasons.append(RoutingReason.PROCESS_IMAGE)

                if PDF_CATEGORY in request_context.prompt_categories and features.can_process_pdf:
                    reasons.append(RoutingReason.PROCESS_PDF)

                if ONLINE_CATEGORY in request_context.prompt_categories and features.is_live:
                    reasons.append(RoutingReason.ONLINE_ACCESS)

                MAX_EXTRA_REASONS = 3

                # Sort criteria by score in descending order and take top 3
                sorted_criteria = sorted(criteria_map.items(), key=lambda x: x[1], reverse=True)
                sorted_criteria = [criteria for criteria in sorted_criteria if criteria[1] > 0]
                added_reasons = 0
                for criteria, _ in sorted_criteria:
                    # Try to map selection criteria to routing reason
                    if criteria == SelectionCriteria.ROUTING_RULES:
                        # TODO(Tian): add routing rule based matching reasons.
                        pass
                    elif criteria in SELECTION_CRITERIA_TO_REASON and added_reasons < MAX_EXTRA_REASONS:
                        reason = SELECTION_CRITERIA_TO_REASON[criteria]
                        if reason not in reasons:
                            reasons.append(reason)
                            added_reasons += 1

        description = None
        if len(reasons) > 0 and reasons[0] in REASON_DESCRIPTION_TEMPLATE:
            description = REASON_DESCRIPTION_TEMPLATE[reasons[0]]

        # generate final summary, the description will be just the first reason
        result[model_internal_name] = ReasonSummary(
            reasons=[r.value.lower() for r in reasons],
            description=description,
        )

    return result
