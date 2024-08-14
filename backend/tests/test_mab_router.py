import random
from collections import Counter
from collections.abc import Generator

import pytest
from mabwiser.mab import LearningPolicy

from backend.llm.constants import MODELS
from backend.llm.mab_router import MABRouter

random.seed(123)


def _get_arm_counts(mab: MABRouter, num_trials: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for _ in range(num_trials):
        arm = mab.select_arms(num_arms=1)[0]
        counter[arm] += 1
    return counter


def _get_decisions_and_rewards(model_rewards: dict[str, float], num_samples: int) -> tuple[list[str], list[float]]:
    decisions = random.choices(
        list(model_rewards.keys()),
        weights=list(model_rewards.values()),
        k=num_samples,
    )
    rewards = [model_rewards[a] for a in decisions]
    return decisions, rewards


@pytest.fixture()
def mab_router() -> Generator[MABRouter, None, None]:
    costs = [10.0, 20.0, 15.0, 25.0, 10.0]
    router = MABRouter(
        arms=MODELS,
        learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.2),
        costs=costs,
    )

    past_actions = [
        ("gpt-4o", 0.8),
        ("gpt-4o-mini", 0.7),
        ("mistral-large-latest", 0.9),
        ("gemini-1.5-pro", 0.7),
        ("claude-3-5-sonnet-20240620", 0.6),
        ("gpt-4o", 0.2),
        ("gpt-4o-mini", 0.3),
        ("mistral-large-latest", 0.1),
        ("gemini-1.5-pro", 0.3),
        ("claude-3-5-sonnet-20240620", 0.4),
    ]
    decisions = [a for a, _ in past_actions]
    rewards = [r for _, r in past_actions]

    router.fit(decisions, rewards)
    yield router


def test_simple_route(mab_router: MABRouter) -> None:
    selected_arms = mab_router.select_arms(num_arms=2)
    assert selected_arms == ["gpt-4o", "gpt-4o-mini"]


def test_budget_too_low(mab_router: MABRouter) -> None:
    with pytest.raises(ValueError):
        mab_router.select_arms(num_arms=2, budget=1)


def test_num_arms_too_high(mab_router: MABRouter) -> None:
    with pytest.raises(ValueError):
        mab_router.select_arms(num_arms=10)


def test_update_model() -> None:
    """Test that the MABRouter can update its model weights."""
    router = MABRouter(
        arms=["chatgpt", "llama"],
        learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.1),
    )
    decisions, rewards = _get_decisions_and_rewards({"chatgpt": 0.9, "llama": 0.1}, 100)
    router.fit(decisions, rewards)
    arm_counts = _get_arm_counts(router, 100)
    assert arm_counts["chatgpt"] > arm_counts["llama"]

    decisions, rewards = _get_decisions_and_rewards({"chatgpt": 0.1, "llama": 0.9}, 200)
    router.update(decisions, rewards)
    arm_counts = _get_arm_counts(router, 100)
    assert arm_counts["chatgpt"] < arm_counts["llama"]

    router.add_arm("claude", 10.0)
    decisions, rewards = _get_decisions_and_rewards({"chatgpt": 0.1, "llama": 0.1, "claude": 0.8}, 100)
    router.update(decisions, rewards)
    arm_counts = _get_arm_counts(router, 100)
    assert arm_counts["claude"] > 1
