"""
Module for classifying review types and routing reviews accordingly.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from ypl.backend.config import settings
from ypl.backend.llm.judge import ReviewRouteClassifier
from ypl.backend.llm.provider.provider_clients import get_provider_client
from ypl.backend.llm.review_types import ReviewRoute
from ypl.backend.llm.routing.router import get_default_routing_llm

# Global model instance for review route classification
REVIEW_ROUTE_CLASSIFIER: ReviewRouteClassifier | None = None

# Model used for review route classification
REVIEW_ROUTE_MODEL = "gpt-4o-mini"


async def get_review_type(prompt: str) -> ReviewRoute:
    """
    Determines whether a prompt should be routed to Pro Review or Cross-Check.
    Only call this when a review is actually needed.

    Returns:
        ReviewRoute: ReviewRoute.PRO or ReviewRoute.CROSS_CHECK indicating the type of review needed
    """
    review_route_classifier = await _get_review_route_classifier()
    review_route_label = await review_route_classifier.alabel(prompt)
    return ReviewRoute("cross_check") if "cross" in review_route_label.lower() else ReviewRoute("pro")


async def get_review_route_classifier_llm(model_name: str | None = None) -> BaseChatModel:
    """
    Get the LLM to use for review route classification.

    Args:
        model_name: Optional model name to use

    Returns:
        BaseChatModel: The LLM to use
    """
    if model_name:
        return await get_provider_client(internal_name=model_name)
    else:
        return await get_default_routing_llm()


async def _get_review_route_classifier() -> ReviewRouteClassifier:
    """
    Get (or create) the review route classifier.

    Returns:
        ReviewRouteClassifier: The classifier instance
    """
    global REVIEW_ROUTE_CLASSIFIER
    if REVIEW_ROUTE_CLASSIFIER is None:
        REVIEW_ROUTE_CLASSIFIER = ReviewRouteClassifier(
            await get_review_route_classifier_llm(REVIEW_ROUTE_MODEL),
            timeout_secs=settings.ROUTING_TIMEOUT_SECS,
        )
    return REVIEW_ROUTE_CLASSIFIER
