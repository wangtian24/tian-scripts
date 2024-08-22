from enum import Enum


class SelectionCriteria(Enum):
    # Select models based on the expected reward.
    TOP = "top"

    # Select models with a probability proportional to their expected reward.
    PROPORTIONAL = "proportional"

    # Select the models with the largest confidence intervals.
    CONF_INTERVAL = "conf_interval"

    RANDOM = "random"


class RoutingPolicy:
    """Policy for selecting models to route to."""

    def __init__(
        self,
        selection_criteria: SelectionCriteria,
        model_traffic_fraction: dict[str, float] | None = None,
    ):
        self.selection_criteria = selection_criteria
        self.model_traffic_fraction = model_traffic_fraction or {}


DEFAULT_ROUTING_POLICY = RoutingPolicy(selection_criteria=SelectionCriteria.TOP)
