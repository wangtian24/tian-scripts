from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ypl.backend.llm.ranking import Ranker


def exponential_decay(initial_value: float, final_value: float, total_steps: int, current_step: int) -> float:
    """Returns a value that decays exponentially from initial_value to final_value over total_steps, at current_step."""
    assert total_steps > 1
    decay_rate = (final_value / initial_value) ** (1 / total_steps)
    return max(final_value, float(initial_value * (decay_rate**current_step)))


def fixed_random_fraction(ranker: "Ranker", value: float) -> float:
    """Returns a fixed fraction."""
    return value


def decayed_random_fraction(ranker: "Ranker", initial_value: float, final_value: float, steps: int) -> float:
    """Returns the exponentially decayed value from `exponential_decay` after `ranker.total_battles` steps."""
    val = exponential_decay(initial_value, final_value, steps, ranker.total_battles)
    return val


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
        minimum_model_traffic_fraction: dict[str, float] | None = None,
        random_fraction: Callable[["Ranker"], float] | float | None = None,
    ):
        """Initialize the routing policy.

        Args:
            selection_criteria: The selection criteria.
            minimum_model_traffic_fraction: Optional minimum traffic fraction for each model.
            random_fraction: A fraction of the traffic that is randomly selected, or a function that takes a `Ranker`
                and returns the fraction of the traffic that is randomly selected.
        """
        self.selection_criteria = selection_criteria
        self.minimum_model_traffic_fraction = minimum_model_traffic_fraction or {}
        if random_fraction is not None:
            if isinstance(random_fraction, float):
                # Convert a float to a function that always returns it.
                random_fraction = partial(fixed_random_fraction, value=random_fraction)
        self.random_fraction = random_fraction


DEFAULT_ROUTING_POLICY = RoutingPolicy(
    selection_criteria=SelectionCriteria.PROPORTIONAL,
    # Decay the fraction of random traffic from 0.6 to 0.05 over 50,000 battles.
    random_fraction=partial(decayed_random_fraction, initial_value=0.6, final_value=0.05, steps=50000),
)
