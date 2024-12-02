from enum import Enum
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
    # These are internal policies.
    _MIN_TRAFFIC_FRACTION = "min_traffic_fraction"  # from the setting MINIMUM_MODEL_TRAFFIC_FRACTION (see router.py)
    _ALWAYS_INCLUDE_TOP = "always_include_top"  # from the setting ROUTING_GOOD_MODELS_ALWAYS (router.py, config.py)

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

    # Select the models with the best quality for the given prompt
    BEST_PROMPT_QUALITY = "best_prompt_quality"

    # Select the models with the fastest speed
    MAX_SPEED = "max_speed"

    # Select the models routed by the prompt categorizer
    PROMPT_CATEGORIZER = "prompt_categorizer"

    # Select the pro models
    PRO_MODELS = "pro_models"

    # Select the models routed by the routing rules
    ROUTING_RULES = "routing_rules"

    RANDOM = "random"
