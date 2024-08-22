from collections import Counter
from typing import Any

from pytest import approx, mark

from backend.llm.ranking import Battle, ChoixRanker, ChoixRankerConfIntervals, EloRanker
from backend.llm.routing.policy import RoutingPolicy, SelectionCriteria
from backend.llm.routing.router import RankedRouter
from backend.tests.utils import get_battles

CONF_INTERVAL_ROUTING_POLICY = RoutingPolicy(SelectionCriteria.CONF_INTERVAL)
PROPORTIONAL_ROUTING_POLICY = RoutingPolicy(SelectionCriteria.PROPORTIONAL)
TOP_ROUTING_POLICY = RoutingPolicy(SelectionCriteria.TOP)


def _check_list_len_distribution(lst: list[Any], expected: dict[int, Any]) -> None:
    counter = Counter([len(sublist) for sublist in lst])
    assert counter == expected


def _check_list_item_distribution(lst: list[Any], expected: dict[str, Any]) -> None:
    counter = Counter([item for sublist in lst for item in sublist])
    assert counter == expected


def test_proportional_routing() -> None:
    models = ["gpt-1", "gpt-2"]
    ranker = EloRanker(models=models, k=10)
    for _ in range(20):
        ranker.update("gpt-1", "gpt-2", 0)

    router = RankedRouter(models, policy=PROPORTIONAL_ROUTING_POLICY, ranker=ranker, seed=11)

    battles = [router.select_models(2) for _ in range(10)]
    _check_list_len_distribution(battles, {2: 10})
    _check_list_item_distribution(battles, {"gpt-1": 10, "gpt-2": 10})

    ranker.add_model("gpt-3")
    for _ in range(20):
        ranker.update("gpt-3", "gpt-1", 1)
    for _ in range(20):
        ranker.update("gpt-3", "gpt-2", 1)

    battles = [router.select_models(2) for _ in range(200)]
    _check_list_len_distribution(battles, {2: 200})
    _check_list_item_distribution(
        battles, {"gpt-1": approx(80, abs=10), "gpt-2": approx(150, abs=10), "gpt-3": approx(180, abs=10)}
    )


def test_top_routing() -> None:
    model_rewards = {"gpt-1": 0.5, "gpt-2": 1.0, "gpt-3": 2.0}
    models = list(model_rewards.keys())
    battles, results = get_battles(model_rewards, 50)
    battles_objs = [Battle(battle[0], battle[1], result) for battle, result in zip(battles, results, strict=True)]
    ranker = ChoixRanker(models, choix_ranker_algorithm="rank_centrality", battles=battles_objs)
    router = RankedRouter(models, policy=TOP_ROUTING_POLICY, ranker=ranker, seed=11)

    routed_battles = [router.select_models(2) for _ in range(10)]
    _check_list_len_distribution(routed_battles, {2: 10})
    _check_list_item_distribution(routed_battles, {"gpt-3": 10, "gpt-2": 10})


def test_decrease_conf_interval_routing() -> None:
    models = ["a", "b", "c", "d"]
    ranker = ChoixRankerConfIntervals(
        models=models,
        num_bootstrap_iterations=5,
        choix_ranker_algorithm="lsr_pairwise",
    )

    # a and b have the same amount of wins/losses against each other, while c always wins over d,
    # so the confidence intervals should be wider for a and b.
    for _ in range(2):
        ranker.update("a", "b", 1.0)
        ranker.update("a", "b", 0.0)
        ranker.update("c", "d", 1.0)
        ranker.update("d", "c", 0.0)

    router = RankedRouter(models, policy=CONF_INTERVAL_ROUTING_POLICY, ranker=ranker, seed=11)
    routed_battles = [router.select_models(2) for _ in range(10)]

    assert sorted(routed_battles[0]) == ["a", "b"]

    # Make the battles less ambiguous for a/b, and more ambiguous for c/d.
    for _ in range(3):
        ranker.update("a", "b", 1.0)
        ranker.update("b", "a", 0.0)
        ranker.update("c", "d", 0.0)
        ranker.update("c", "d", 1.0)

    routed_battles = [router.select_models(2) for _ in range(10)]
    assert sorted(routed_battles[0]) == ["c", "d"]


def test_traffic_fraction_routing() -> None:
    models = ["a", "b", "c", "d"]
    ranker = ChoixRanker(
        models=models,
        choix_ranker_algorithm="lsr_pairwise",
    )
    policy = RoutingPolicy(SelectionCriteria.RANDOM, minimum_model_traffic_fraction={"c": 0.3, "d": 0.4})
    router = RankedRouter(models, policy=policy, ranker=ranker, seed=11)

    selected_by_fraction = [router._select_models_by_minimum_traffic_fraction(3) for _ in range(1000)]
    # Ensure reasonable distribution of the number of selected models and the models selected.
    expected_lengths = {2: approx(110, rel=0.1), 1: approx(450, rel=0.1), 0: approx(450, rel=0.1)}
    _check_list_len_distribution(selected_by_fraction, expected_lengths)
    _check_list_item_distribution(selected_by_fraction, {"c": approx(300, rel=0.1), "d": approx(400, rel=0.1)})

    battles = [router.select_models(2) for _ in range(1000)]

    # All battles should have 2 different models.
    _check_list_len_distribution([tuple(set(battle)) for battle in battles], {2: 1000})

    # Model c should be > 30% of the time, and model d > 40%.
    # Count the occurrences of each model in the battles
    expected_distribution = {
        "c": approx(600, rel=0.2),
        "d": approx(700, rel=0.2),
        "a": approx(350, rel=0.2),
        "b": approx(350, rel=0.2),
    }

    _check_list_item_distribution(battles, expected_distribution)


@mark.parametrize(
    "random_fraction, expected_distribution",
    [
        (
            0.0,
            {
                # No random fraction:
                #   a gets a lot of traffic as the highest-ranked model.
                #   b gets no traffic as the lowest-ranked model.
                #   c and d get equal amounts of traffic, lower than a.
                "a": approx(1000, rel=0.1),
                "c": approx(500, rel=0.1),
                "d": approx(500, rel=0.1),
            },
        ),
        (
            0.5,
            {
                # 50% random fraction:
                #   a gets a lot of traffic as the highest-ranked model.
                #   b gets some traffic.
                #   c and d get equal amounts of traffic, lower than a, but higher than b.
                "a": approx(900, rel=0.2),
                "b": approx(250, rel=0.2),
                "c": approx(425, rel=0.2),
                "d": approx(425, rel=0.2),
            },
        ),
        (
            1.0,
            {
                # 100% random fraction: all models get equal amounts of traffic.
                "a": approx(500, rel=0.1),
                "b": approx(500, rel=0.1),
                "c": approx(500, rel=0.1),
                "d": approx(500, rel=0.1),
            },
        ),
    ],
)
def test_random_routing(random_fraction: float, expected_distribution: dict[str, float]) -> None:
    models = ["a", "b", "c", "d"]
    policy = RoutingPolicy(SelectionCriteria.TOP, random_fraction=random_fraction)
    ranker = ChoixRanker(models=models, choix_ranker_algorithm="lsr_pairwise")
    ranker.update("a", "b", 1.0)
    ranker.update("a", "b", 1.0)
    router = RankedRouter(models, policy=policy, ranker=ranker, seed=11)

    selected = [router.select_models(2) for _ in range(1000)]
    _check_list_len_distribution(selected, {2: 1000})
    _check_list_item_distribution(selected, expected_distribution)
