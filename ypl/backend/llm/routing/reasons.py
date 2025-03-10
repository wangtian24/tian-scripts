from enum import Enum

from pydantic import BaseModel
from sqlmodel import select

from ypl.backend.db import get_async_session
from ypl.backend.llm.constants import IMAGE_CATEGORY, IMAGE_GEN_CATEGORY, ONLINE_CATEGORY, PDF_CATEGORY
from ypl.backend.llm.routing.common import SelectIntent
from ypl.backend.llm.routing.features import ModelSetFeatures, RequestContext
from ypl.backend.llm.routing.policy import SelectionCriteria
from ypl.backend.llm.routing.router_state import RouterState
from ypl.db.reasons import RoutingReason
from ypl.utils import async_timed_cache


class RoutingReasonType(Enum):
    UNKNOWN = "unknown"
    USER_SELECTED = "user_selected"
    ONLINE_ACCESS = "online_access"
    PROCESS_IMAGE = "process_image"
    PROCESS_PDF = "process_pdf"
    CODING = "coding"
    IMAGE_GENERATION = "image_generation"
    PRO_OR_STRONG = "pro_or_strong"
    REASONING = "reasoning"
    FAST = "fast"


# mapping from (some) selection criteria to routing reasons. right now not all reasons are easily mappable
# from SelectionCriteria, like ROUTING_RULES can be for coding/online access and others.
# This is ordered by the priority in which we want to show them.
SELECTION_CRITERIA_TO_REASON = {
    SelectionCriteria.IMAGE_GENERATION_MODELS: RoutingReasonType.IMAGE_GENERATION,
    SelectionCriteria.REASONING_MODELS: RoutingReasonType.REASONING,
    SelectionCriteria.LIVE_MODELS: RoutingReasonType.ONLINE_ACCESS,
    SelectionCriteria.PRO_MODELS: RoutingReasonType.PRO_OR_STRONG,
    SelectionCriteria.STRONG_MODELS: RoutingReasonType.PRO_OR_STRONG,
    SelectionCriteria.PRO_AND_STRONG_MODELS: RoutingReasonType.PRO_OR_STRONG,
    SelectionCriteria.FAST_MODELS: RoutingReasonType.FAST,
}

MAX_EXTRA_REASONS = 3


class ReasonSummary(BaseModel):
    reasons: list[str]
    description: str | None


@async_timed_cache(seconds=600)
async def _get_routing_reason_descriptions() -> dict[str, str]:
    """
    Get all routing reason descriptions from DB.
    Returns a dictionary of reason types to descriptions, all keys in lower
    """
    query = select(RoutingReason).where(RoutingReason.is_active.is_(True), RoutingReason.deleted_at.is_(None))  # type: ignore
    async with get_async_session() as session:
        return {r.reason.lower(): r.description for r in (await session.exec(query)).all()}  # type: ignore


async def summarize_reasons(
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
        reasons: list[RoutingReasonType] = []
        features = model_set_features.model_features[model_internal_name]

        if (
            model_internal_name in request_context.user_required_models
            and request_context.intent == SelectIntent.NEW_CHAT
        ):
            # user selected model, will not consider other reasons
            reasons.append(RoutingReasonType.USER_SELECTED)
        elif model_internal_name not in request_context.inherited_models:
            # match against model capabilities
            if IMAGE_CATEGORY in request_context.prompt_categories and features.can_process_image:
                reasons.append(RoutingReasonType.PROCESS_IMAGE)

            if PDF_CATEGORY in request_context.prompt_categories and features.can_process_pdf:
                reasons.append(RoutingReasonType.PROCESS_PDF)

            if IMAGE_GEN_CATEGORY in request_context.prompt_categories and features.is_image_generation:
                reasons.append(RoutingReasonType.IMAGE_GENERATION)

            # TODO(Tian): Adding coding reason inference using the information from routing table.
            # if CODING_CATEGORY in request_context.prompt_categories:
            #     reasons.append(RoutingReasonType.CODING)

            if ONLINE_CATEGORY in request_context.prompt_categories and features.is_live:
                reasons.append(RoutingReasonType.ONLINE_ACCESS)

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
        reason_descriptions = await _get_routing_reason_descriptions()

        # generate final summary, just generate the description for the first reason.
        if len(reasons) > 0 and reasons[0].value in reason_descriptions:
            description = reason_descriptions[reasons[0].value]

        result[model_internal_name] = ReasonSummary(
            reasons=[r.value.lower() for r in reasons],
            description=description,
        )

    return result
