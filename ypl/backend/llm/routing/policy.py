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
    TOP_K = "top_k"
    TOP_ELO = "top_elo"

    # Select models with a probability proportional to their expected reward.
    PROPORTIONAL = "proportional"

    # -- Confidence interval related
    # Select the models with the largest confidence intervals.
    CONF_INTERVAL_WIDTH = "conf_interval"
    # Select the models with the largest overlap of their confidence intervals.
    CONF_INTERVAL_NUM_OVERLAP = "conf_interval_num_overlap"
    # Select the models with the greatest pair overlap of their CIs.
    CONF_INTERVAL_PAIR_OVERLAP = "conf_interval_pair_overlap"

    # -- Running cost related
    # Select the models with the lowest running cost.
    MIN_RUNNING_COST = "min_running_cost"
    # Select the models with the highest running cost.
    MAX_RUNNING_COST = "max_running_cost"
    # Select the models with the lowest cost per million tokens.
    MIN_SIMPLE_COST = "min_simple_cost"
    # Select the models with the highest cost per million tokens.
    MAX_SIMPLE_COST = "max_simple_cost"

    # -- Quality related
    # Select the models with the best quality for the given prompt
    BEST_PROMPT_QUALITY = "best_prompt_quality"

    # -- Speed related
    # Select the models with the fastest speed
    MAX_SPEED = "max_speed"

    # -- Prompt categorizer related
    # Select the models routed by the prompt categorizer
    PROMPT_CATEGORIZER = "prompt_categorizer"

    # -- Model designation related
    PRO_MODELS = "pro_models"
    STRONG_MODELS = "strong_models"
    PRO_AND_STRONG_MODELS = "pro_and_strong_models"
    FAST_MODELS = "fast_models"
    LIVE_MODELS = "live_models"
    REASONING_MODELS = "reasoning_models"

    # -- Attachment related
    IMAGE_MODELS = "image_models"
    PDF_MODELS = "pdf_models"

    # -- Model promotion related
    PROMOTED_MODELS = "promoted_models"

    # Select the models routed by the routing rules
    ROUTING_RULES = "routing_rules"

    # Select the models injected by the user or some process
    INJECT = "inject"

    # -- Random related
    RANDOM = "random"
    RANDOM_REPUTABLE = "random_reputable"
    RANDOM_SHUFFLE = "random_shuffle"
