import random
from collections import Counter
from collections.abc import Generator

import pytest
from mabwiser.mab import LearningPolicy

from backend.llm.mab_ranker import MultiArmedBanditRanker
from backend.llm.routing.policy import RoutingPolicy, SelectionCriteria
from backend.llm.routing.router import RankedRouter
from backend.tests.utils import get_battles

random.seed(123)

ROUTING_POLICY = RoutingPolicy(selection_criteria=SelectionCriteria.TOP)


def _get_model_counts(router: RankedRouter, num_trials: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for _ in range(num_trials):
        model = router.select_models(num_models=1)[0]
        counter[model] += 1
    return counter


@pytest.fixture()
def router() -> Generator[RankedRouter, None, None]:
    costs = [10.0, 20.0, 15.0, 25.0, 10.0]
    models = ["gpt-4o", "gpt-4o-mini", "mistral-large-latest", "gemini-1.5-pro", "claude-3-5-sonnet-20240620"]
    ranker = MultiArmedBanditRanker(
        models=models,
        learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.2),
        costs=costs,
    )

    past_actions = [
        (("gpt-4o", "gemini-1.5-pro"), 0.8),
        (("gpt-4o-mini", "mistral-large-latest"), 0.7),
        (("mistral-large-latest", "gpt-4o-mini"), 0.9),
        (("gemini-1.5-pro", "gpt-4o"), 0.7),
        (("claude-3-5-sonnet-20240620", "gpt-4o"), 0.6),
        (("gpt-4o", "gpt-4o-mini"), 0.2),
        (("gpt-4o-mini", "mistral-large-latest"), 0.3),
        (("mistral-large-latest", "gpt-4o-mini"), 0.1),
        (("gemini-1.5-pro", "gpt-4o"), 0.3),
        (("claude-3-5-sonnet-20240620", "gpt-4o"), 0.4),
    ]
    battles = [a for a, _ in past_actions]
    rewards = [r for _, r in past_actions]

    ranker.fit(battles, rewards)
    router = RankedRouter(models=models, ranker=ranker, policy=ROUTING_POLICY)
    yield router


def test_simple_route(router: RankedRouter) -> None:
    routes = []
    for _ in range(200):
        selected = router.select_models(num_models=2)
        routes.append(tuple(sorted(selected)))
    common_routes = [r for r, _ in Counter(routes).most_common(5)]
    for expected_common_route in [
        ("gpt-4o", "gpt-4o-mini"),
        ("gpt-4o-mini", "mistral-large-latest"),
        ("claude-3-5-sonnet-20240620", "gpt-4o-mini"),
    ]:
        assert expected_common_route in common_routes


def test_budget_too_low(router: RankedRouter) -> None:
    with pytest.raises(ValueError):
        router.select_models(num_models=2, budget=1)


def test_num_models_too_high(router: RankedRouter) -> None:
    with pytest.raises(ValueError):
        router.select_models(num_models=10)


def test_update_model() -> None:
    """Test that the MABRouter can update its model weights."""
    models = ["chatgpt", "llama"]
    ranker = MultiArmedBanditRanker(
        models=models,
        learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.1),
    )
    router = RankedRouter(
        models=models,
        policy=ROUTING_POLICY,
        ranker=ranker,
    )
    battles, rewards = get_battles({"chatgpt": 0.9, "llama": 0.1}, 100)
    ranker.fit(battles, rewards)
    model_counts = _get_model_counts(router, 100)
    assert model_counts["chatgpt"] > model_counts["llama"]

    battles, rewards = get_battles({"chatgpt": 0.1, "llama": 0.9}, 200)
    ranker.update_batch(battles, rewards)
    model_counts = _get_model_counts(router, 100)
    assert model_counts["chatgpt"] < model_counts["llama"]

    ranker.add_model("claude", 10.0)
    battles, rewards = get_battles({"chatgpt": 0.1, "llama": 0.1, "claude": 0.8}, 100)
    ranker.update_batch(battles, rewards)
    model_counts = _get_model_counts(router, 100)
    assert model_counts["claude"] > 1
