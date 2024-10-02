import random
from collections import Counter
from collections.abc import Generator

import pytest
from mabwiser.mab import LearningPolicy

from ypl.backend.llm.mab_ranker import MultiArmedBanditRanker
from ypl.backend.llm.routing.router import EloProposer, RouterModule, RouterState
from ypl.backend.tests.utils import get_battles

random.seed(123)


def _get_model_counts(router: RouterModule, num_trials: int, models: list[str]) -> dict[str, int]:
    counter: Counter[str] = Counter()

    for _ in range(num_trials):
        model = router.select_models(1, state=RouterState(all_models=set(models))).get_sorted_selected_models()[0]
        counter[model] += 1

    return counter


@pytest.fixture()
def router_and_models() -> Generator[tuple[RouterModule, list[str]], None, None]:
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
    router = EloProposer(ranker)

    yield router, models


def test_simple_route(router_and_models: tuple[RouterModule, list[str]]) -> None:
    router, models = router_and_models
    routes = []

    for _ in range(200):
        selected = list(router.select_models(2, state=RouterState(all_models=set(models))).get_selected_models())
        routes.append(tuple(sorted(selected)))

    common_routes = [r for r, _ in Counter(routes).most_common(5)]

    for expected_common_route in [
        ("gpt-4o", "gpt-4o-mini"),
        ("gpt-4o-mini", "mistral-large-latest"),
        ("claude-3-5-sonnet-20240620", "gpt-4o-mini"),
    ]:
        assert expected_common_route in common_routes


def test_update_model() -> None:
    """Test that the MABRouter can update its model weights."""
    models = ["chatgpt", "llama"]
    ranker = MultiArmedBanditRanker(
        models=models,
        learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.1),
    )

    router = EloProposer(ranker)
    battles, rewards = get_battles({"chatgpt": 0.9, "llama": 0.1}, 100)
    ranker.fit(battles, rewards)
    model_counts = _get_model_counts(router, 100, models)
    assert model_counts["chatgpt"] > model_counts["llama"]

    battles, rewards = get_battles({"chatgpt": 0.1, "llama": 0.9}, 200)
    ranker.update_batch(battles, rewards)
    model_counts = _get_model_counts(router, 100, models)
    assert model_counts["chatgpt"] < model_counts["llama"]

    ranker.add_model("claude", 10.0)
    battles, rewards = get_battles({"chatgpt": 0.1, "llama": 0.1, "claude": 0.8}, 100)
    ranker.update_batch(battles, rewards)
    model_counts = _get_model_counts(router, 100, models + ["claude"])
    assert model_counts["claude"] > 1
