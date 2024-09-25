import random
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
    val = exponential_decay(initial_value, final_value, steps, ranker.get_total_battles())
    return val


class SelectionCriteria(Enum):
    # These are for internal logging use only.
    _MIN_TRAFFIC_FRACTION = "min_traffic_fraction"

    # Select models based on the expected reward.
    TOP = "top"

    # Select models with a probability proportional to their expected reward.
    PROPORTIONAL = "proportional"

    # Select the models with the largest confidence intervals.
    CONF_INTERVAL_WIDTH = "conf_interval"

    # Select the models with the largest overlap of their confidence intervals.
    CONF_INTERVAL_NUM_OVERLAP = "conf_interval_num_overlap"

    # Select the models with the greatest pair overlap of their CIs.
    CONF_INTERVAL_PAIR_OVERLAP = "conf_interval_pair_overlap"

    # Select the models with the lowest running cost.
    MIN_RUNNING_COST = "min_running_cost"

    # Select the models with the highest running cost.
    MAX_RUNNING_COST = "max_running_cost"

    # Select the models with the lowest cost per million tokens.
    MIN_SIMPLE_COST = "min_simple_cost"

    # Select the models with the highest cost per million tokens.
    MAX_SIMPLE_COST = "max_simple_cost"

    RANDOM = "random"


class RoutingPolicy:
    """Policy for selecting models to route to."""

    def __init__(
        self,
        selection_criteria: SelectionCriteria | dict[SelectionCriteria, float],
        minimum_model_traffic_fraction: dict[str, float] | None = None,
        random_fraction: Callable[["Ranker"], float] | float | None = None,
        seed: int | None = None,
    ):
        """Initialize the routing policy.

        Args:
            selection_criteria: The selection criteria. If a dict, it is treated as a mapping from selection criteria to
                their weights for random sampling. Otherwise, it is treated as the selection criteria.
            minimum_model_traffic_fraction: Optional minimum traffic fraction for each model.
            random_fraction: A fraction of the traffic that is randomly selected, or a function that takes a `Ranker`
                and returns the fraction of the traffic that is randomly selected.
        """
        self.seed = seed or random.randint(0, 2**32)
        self._rng = random.Random(self.seed)
        self._selection_criteria = selection_criteria
        self.minimum_model_traffic_fraction = minimum_model_traffic_fraction or {}
        if random_fraction is not None:
            if isinstance(random_fraction, float):
                # Convert a float to a function that always returns it.
                random_fraction = partial(fixed_random_fraction, value=random_fraction)
        self.random_fraction = random_fraction

    @property
    def selection_criteria(self) -> SelectionCriteria:
        if isinstance(self._selection_criteria, dict):
            return self._rng.choices(
                list(self._selection_criteria.keys()), weights=list(self._selection_criteria.values())
            )[0]
        else:
            return self._selection_criteria


DEFAULT_ROUTING_POLICY = RoutingPolicy(
    selection_criteria=SelectionCriteria.PROPORTIONAL,
    # Decay the fraction of random traffic from 0.6 to 0.05 over 50,000 battles.
    random_fraction=partial(decayed_random_fraction, initial_value=0.6, final_value=0.05, steps=50000),
)
