from collections import Counter

from backend.llm.ranking import Battle, ChoixRanker, ChoixRankerConfIntervals, EloRanker
from backend.llm.routing import RankedRouter, RoutingPolicy
from backend.tests.utils import get_battles


def _get_model_counts_in_battles(battles: list[tuple[str, str]]) -> dict[str, int]:
    return Counter([model for battle in battles for model in battle])


def test_proportional_routing() -> None:
    models = ["gpt-1", "gpt-2"]
    ranker = EloRanker(models, k=10)
    for _ in range(20):
        ranker.update("gpt-1", "gpt-2", 0)

    router = RankedRouter(models, ranker, seed=11)

    battles = [router.select_models(2, policy=RoutingPolicy.PROPORTIONAL) for _ in range(10)]
    counts = _get_model_counts_in_battles(battles)  # type: ignore
    assert len(counts) == 2
    assert counts["gpt-1"] == counts["gpt-2"] == 10

    ranker.add_model("gpt-3")
    for _ in range(20):
        ranker.update("gpt-3", "gpt-1", 1)
    for _ in range(20):
        ranker.update("gpt-3", "gpt-2", 1)

    battles = [router.select_models(2, policy=RoutingPolicy.PROPORTIONAL) for _ in range(200)]
    counts = _get_model_counts_in_battles(battles)  # type: ignore
    assert len(counts) == 3
    assert counts["gpt-1"] < counts["gpt-2"]
    assert counts["gpt-2"] < counts["gpt-3"]


def test_top_routing() -> None:
    model_rewards = {"gpt-1": 0.5, "gpt-2": 1.0, "gpt-3": 2.0}
    models = list(model_rewards.keys())
    battles, results = get_battles(model_rewards, 50)
    battles_objs = [Battle(battle[0], battle[1], result) for battle, result in zip(battles, results, strict=True)]
    ranker = ChoixRanker(models, choix_ranker_algorithm="rank_centrality", battles=battles_objs)
    router = RankedRouter(models, ranker)

    routed_battles = [router.select_models(2, policy=RoutingPolicy.TOP) for _ in range(10)]
    counts = _get_model_counts_in_battles(routed_battles)  # type: ignore
    assert len(counts) == 2
    assert counts["gpt-3"] == counts["gpt-2"]


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

    router = RankedRouter(models, ranker)
    routed_battles = [router.select_models(2, policy=RoutingPolicy.DECREASE_CONF_INTERVAL) for _ in range(10)]

    assert sorted(routed_battles[0]) == ["a", "b"]

    # Make the battles less ambiguous for a/b, and more ambiguous for c/d.
    for _ in range(3):
        ranker.update("a", "b", 1.0)
        ranker.update("b", "a", 0.0)
        ranker.update("c", "d", 0.0)
        ranker.update("c", "d", 1.0)

    routed_battles = [router.select_models(2, policy=RoutingPolicy.DECREASE_CONF_INTERVAL) for _ in range(10)]
    assert sorted(routed_battles[0]) == ["c", "d"]
