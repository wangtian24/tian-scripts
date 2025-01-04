import asyncio
from collections import Counter
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
from pytest import approx, mark

from ypl.backend.config import settings
from ypl.backend.llm.ranking import Battle, ChoixRanker, ChoixRankerConfIntervals, EloRanker
from ypl.backend.llm.routing.policy import (
    exponential_decay,
)
from ypl.backend.llm.routing.router import (
    AlwaysGoodModelMetaRouter,
    ConfidenceIntervalWidthModelProposer,
    EloProposer,
    MinimumFractionModelProposer,
    ProportionalModelProposer,
    RandomModelProposer,
    RouterState,
    TopK,
    _fast_compute_all_conf_overlap_diffs,
    _fast_compute_all_num_intersections,
    get_simple_pro_router,
)
from ypl.backend.llm.routing.rule_router import RoutingTable
from ypl.backend.tests.utils import get_battles
from ypl.db.language_models import RoutingAction, RoutingRule

settings.ROUTING_DO_LOGGING = False


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

    router = ProportionalModelProposer(2, ranker).with_seed(0)
    starting_state = RouterState(all_models=set(models))

    battles = [list(router.select_models(state=starting_state.deepcopy()).get_selected_models()) for _ in range(10)]
    _check_list_len_distribution(battles, {2: 10})
    _check_list_item_distribution(battles, {"gpt-1": 10, "gpt-2": 10})

    ranker.add_model("gpt-3")
    for _ in range(20):
        ranker.update("gpt-3", "gpt-1", 1)
    for _ in range(20):
        ranker.update("gpt-3", "gpt-2", 1)

    starting_state = RouterState(all_models={"gpt-1", "gpt-2", "gpt-3"})
    battles = [list(router.select_models(state=starting_state.deepcopy()).get_selected_models()) for _ in range(200)]
    _check_list_len_distribution(battles, {2: 200})
    _check_list_item_distribution(
        battles, {"gpt-1": approx(80, abs=20), "gpt-2": approx(150, abs=20), "gpt-3": approx(180, abs=20)}
    )


def test_top_routing() -> None:
    model_rewards = {"gpt-1": 0.5, "gpt-2": 1.0, "gpt-3": 2.0}
    models = list(model_rewards.keys())
    battles, results = get_battles(model_rewards, 50)
    battles_objs = [Battle(battle[0], battle[1], result) for battle, result in zip(battles, results, strict=True)]
    ranker = ChoixRanker(models, choix_ranker_algorithm="rank_centrality", battles=battles_objs)
    router = EloProposer(ranker) | TopK(2)
    starting_state = RouterState(all_models=set(models))

    routed_battles = [
        list(router.select_models(state=starting_state.deepcopy()).get_selected_models()) for _ in range(10)
    ]
    _check_list_len_distribution(routed_battles, {2: 10})
    _check_list_item_distribution(routed_battles, {"gpt-3": 10, "gpt-2": 10})


def test_always_include_top_routing() -> None:
    model_rewards = {"gpt-0": 0.25, "gpt-1": 0.5, "gpt-2": 1.0, "gpt-3": 100.0}
    models = list(model_rewards.keys())
    battles, results = get_battles(model_rewards, 1000)
    battles_objs = [Battle(battle[0], battle[1], result) for battle, result in zip(battles, results, strict=True)]
    starting_state = RouterState(all_models=set(models))

    ranker = ChoixRanker(models, choix_ranker_algorithm="rank_centrality", battles=battles_objs)
    router = AlwaysGoodModelMetaRouter(ranker, RandomModelProposer().with_seed(0), num_good=1) | TopK(2)

    routed_battles = [
        list(router.select_models(state=starting_state.deepcopy()).get_selected_models()) for _ in range(100)
    ]

    _check_list_item_distribution(
        routed_battles,
        {
            "gpt-3": approx(100, abs=20),
            "gpt-2": approx(30, abs=15),
            "gpt-1": approx(30, abs=15),
            "gpt-0": approx(30, abs=15),
        },
    )


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

    router = ConfidenceIntervalWidthModelProposer(2, ranker)
    starting_state = RouterState(all_models=set(models))
    routed_battles = [
        list(router.select_models(state=starting_state.deepcopy()).get_selected_models()) for _ in range(10)
    ]

    assert sorted(routed_battles[0]) == ["a", "b"]

    # Make the battles less ambiguous for a/b, and more ambiguous for c/d.
    for _ in range(3):
        ranker.update("a", "b", 1.0)
        ranker.update("b", "a", 0.0)
        ranker.update("c", "d", 0.0)
        ranker.update("c", "d", 1.0)

    routed_battles = [
        list(router.select_models(state=starting_state.deepcopy()).get_selected_models()) for _ in range(10)
    ]
    assert sorted(routed_battles[0]) == ["c", "d"]


def test_traffic_fraction_routing() -> None:
    models = ["a", "b", "c", "d"]
    router = (
        MinimumFractionModelProposer(2, {"c": 0.3, "d": 0.4}).with_seed(0) ^ RandomModelProposer().with_seed(0)
    ).with_probs(0.4, 0.6).with_seed(0) | TopK(2)
    starting_state = RouterState(all_models=set(models))
    battles = [list(router.select_models(state=starting_state.deepcopy()).get_selected_models()) for _ in range(1000)]

    # All battles should have 2 different models.
    _check_list_len_distribution([tuple(set(battle)) for battle in battles], {2: 1000})

    # Model c should be > 30% of the time, and model d > 40%.
    # Count the occurrences of each model in the battles
    expected_distribution = {
        "c": approx(600, rel=0.25),
        "d": approx(700, rel=0.25),
        "a": approx(350, rel=0.25),
        "b": approx(350, rel=0.25),
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
                "a": approx(750, rel=0.2),
                "b": approx(250, rel=0.2),
                "c": approx(500, rel=0.2),
                "d": approx(500, rel=0.2),
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
    ranker = ChoixRanker(models=models, choix_ranker_algorithm="lsr_pairwise")
    ranker.update("a", "b", 1.0)
    ranker.update("a", "b", 1.0)
    router = (EloProposer(ranker).with_seed(0) ^ RandomModelProposer().with_seed(0)).with_probs(
        1 - random_fraction, random_fraction
    ).with_seed(0) | TopK(2)

    selected = [
        list(router.select_models(state=RouterState(all_models=set(models))).get_selected_models()) for _ in range(1000)
    ]

    _check_list_len_distribution(selected, {2: 1000})
    _check_list_item_distribution(selected, expected_distribution)


def test_exponential_decay() -> None:
    decays = [
        exponential_decay(initial_value=1, final_value=0.1, total_steps=100, current_step=i) for i in range(0, 100, 10)
    ]
    assert [round(d, 3) for d in decays] == [1.0, 0.794, 0.631, 0.501, 0.398, 0.316, 0.251, 0.2, 0.158, 0.126]

    decays = [
        exponential_decay(initial_value=1, final_value=0.1, total_steps=10, current_step=i) for i in range(0, 25, 5)
    ]
    assert [round(d, 3) for d in decays] == [1.0, 0.316, 0.1, 0.1, 0.1]


def test_fast_compute_all_num_intersections() -> None:
    intervals = np.array([[0, 1.0], [5.0, 10.0], [2.0, 8.0], [9.0, 15.0], [12.0, 20.0]])
    counts, permutation_map = _fast_compute_all_num_intersections(intervals)
    assert counts.tolist() == [0, 1, 2, 2, 1]
    assert permutation_map.tolist() == [0, 2, 1, 3, 4]

    intervals = np.array([[5.0, 10.0], [0, 6.0]])
    counts, permutation_map = _fast_compute_all_num_intersections(intervals)
    assert counts.tolist() == [1, 1]
    assert permutation_map.tolist() == [1, 0]


def test_fast_compute_all_conf_overlap_diffs() -> None:
    intervals = np.array([[0, 1.0], [3.0, 10.0], [2.0, 8.0], [9.0, 15.0], [12.0, 20.0]])
    inds, vals = _fast_compute_all_conf_overlap_diffs(intervals, k=1)
    assert inds.tolist() == [[1, 2]]
    assert vals.tolist() == approx([5.0])

    intervals = np.array([[3.0, 4.0], [9.0, 15.0]])
    inds, vals = _fast_compute_all_conf_overlap_diffs(intervals, k=100)
    assert inds.tolist() == [[0, 1]]
    assert vals.tolist() == approx([0.0])


@patch("ypl.backend.llm.routing.router.YuppMultilabelClassifier.alabel")
@patch("ypl.backend.llm.routing.router.YuppOnlinePromptLabeler.alabel")
@patch("ypl.backend.llm.routing.router.HighErrorRateFilter.select_models")
@patch("ypl.backend.llm.routing.router.get_all_strong_models")
@patch("ypl.backend.llm.routing.router.get_all_pro_models")
@patch("ypl.backend.llm.routing.router.deduce_original_providers")
@patch("ypl.backend.llm.routing.rule_router.deduce_original_providers")
@patch("ypl.backend.llm.routing.router.deduce_semantic_groups")
@patch("ypl.backend.llm.routing.rule_router.get_routing_table")
@pytest.mark.asyncio
async def test_simple_pro_router(
    mock_routing_table: Mock,
    mock_semantic_group_map: Mock,
    mock_deduce_providers1: Mock,
    mock_deduce_providers2: Mock,
    mock_get_all_pro_models: Mock,
    mock_get_all_strong_models: Mock,
    mock_error_filter: Mock,
    mock_alabel: Mock,
    mock_topic_categorizer: Mock,
) -> None:
    # Test that we get different models
    mock_alabel.return_value = asyncio.Future()
    mock_alabel.return_value.set_result("advice")
    mock_topic_categorizer.return_value = []
    mock_routing_table.return_value = RoutingTable([])
    pro_models = {"pro1", "pro2", "pro3", "pro4"}
    mock_get_all_pro_models.return_value = pro_models
    mock_semantic_group_map.return_value = {}

    models = {"model1", "model2"}
    reputable_providers = {"pro1", "pro2", "pro3", "model1"}
    all_models = models | pro_models
    state = RouterState(all_models=all_models)
    # Just make a provider for each model named after the model.
    mock_deduce_providers1.return_value = {model: model for model in all_models}
    mock_deduce_providers2.return_value = {model: model for model in all_models}
    mock_get_all_strong_models.return_value = {"pro1", "pro2", "pro3", "model1"}
    mock_error_filter.side_effect = lambda state: state

    all_selected_models = set()
    for _ in range(50):
        router = await get_simple_pro_router(prompt="", num_models=2, reputable_providers=reputable_providers)
        selected_models = router.select_models(state=state).get_sorted_selected_models()
        assert len(selected_models) == 2
        assert (selected_models[0] in reputable_providers) or (selected_models[1] in reputable_providers)
        assert selected_models[0] != selected_models[1]
        all_selected_models.update(selected_models)

    # Over all iterations, all reputable or pro models should be selected at least once.
    assert all_selected_models == reputable_providers | pro_models

    # Test that we get the same models if we specify them.
    for _ in range(10):
        router = await get_simple_pro_router(
            prompt="", num_models=2, reputable_providers=reputable_providers, user_selected_models=["model1", "model2"]
        )
        selected_models = router.select_models(state=state).get_sorted_selected_models()
        assert selected_models == ["model1", "model2"]

    for _ in range(10):
        router = await get_simple_pro_router(
            prompt="", num_models=1, reputable_providers=reputable_providers, user_selected_models=["model1"]
        )
        selected_models = router.select_models(state=state).get_sorted_selected_models()
        assert "model1" in selected_models

    # Test that semantic group filtering works.
    models = {"model1", "model2", "model3", "model4", "model5"}
    reputable_providers = {"pro1", "pro2", "pro3", "model1"}
    all_models = models | pro_models
    state = RouterState(all_models=all_models)
    # Just make a provider for each model named after the model.
    mock_deduce_providers1.return_value = {model: model for model in all_models}
    mock_deduce_providers2.return_value = {model: model for model in all_models}
    mock_get_all_strong_models.return_value = {"pro1", "pro2", "pro3", "model1"}
    mock_error_filter.side_effect = lambda state: state
    mock_semantic_group_map.return_value = {"model1": "group1", "model2": "group1"}

    router = await get_simple_pro_router(
        prompt="", num_models=1, reputable_providers=reputable_providers, user_selected_models=["model2"]
    )

    for _ in range(15):
        state = RouterState(all_models=models)
        selected_models = router.select_models(state=state).get_sorted_selected_models()
        assert "model2" in selected_models and "model1" not in selected_models

    router_two_selected_models = await get_simple_pro_router(
        prompt="", num_models=2, reputable_providers=reputable_providers, user_selected_models=["model2", "model1"]
    )
    for _ in range(5):
        state = RouterState(all_models=models)
        selected_models = router_two_selected_models.select_models(state=state).get_sorted_selected_models()
        assert "model2" in selected_models and "model1" in selected_models


@patch("ypl.backend.llm.routing.rule_router.deduce_original_providers")
def test_routing_table(mock_deduce_providers: Mock) -> None:
    mock_deduce_providers.return_value = {
        "model1": "provider1",
        "model2": "provider1",
        "model3": "provider2",
    }

    # Two categories, the catch-all "*" and "advice".
    routing_table = RoutingTable(
        [
            RoutingRule(source_category="*", destination="provider1/model1", target=RoutingAction.ACCEPT, z_index=1000),
            RoutingRule(source_category="*", destination="provider1/*", target=RoutingAction.REJECT, z_index=10),
            RoutingRule(source_category="advice", destination="provider2/*", target=RoutingAction.ACCEPT, z_index=200),
            RoutingRule(source_category="advice", destination="provider1/*", target=RoutingAction.REJECT, z_index=100),
        ]
    )

    # Accept 1 and 3; reject 2
    accept_map, rejected_models = routing_table.apply(("*",), {"model1", "model2", "model3"})
    assert accept_map == {"model1": approx(1000.0), "model3": approx(0.0)}
    assert rejected_models == {"model2"}

    # Accept 1 with score 1k and 3 with score 200; reject 2
    accept_map, rejected_models = routing_table.apply(("advice",), {"model1", "model2", "model3"})
    assert accept_map == {"model1": approx(1000.0), "model3": approx(200.0)}
    assert rejected_models == {"model2"}

    # One category with a globbed model name
    mock_deduce_providers.return_value = {
        "model1-1": "provider1",
        "model1-2": "provider1",
        "model2": "provider1",
        "model3": "provider2",
    }

    routing_table = RoutingTable(
        [
            RoutingRule(
                source_category="*", destination="provider1/model1*", target=RoutingAction.ACCEPT, z_index=1000
            ),
            RoutingRule(source_category="*", destination="provider2/model3", target=RoutingAction.REJECT, z_index=10),
        ]
    )

    accept_map, rejected_models = routing_table.apply(("*",), {"model1-1", "model1-2", "model2", "model3"})

    # Accept no-op model2 (score of 0), and model1-1 and model1-2 with score 1k.
    assert accept_map == {"model1-1": approx(1000.0), "model1-2": approx(1000.0), "model2": approx(0.0)}
    assert rejected_models == {"model3"}

    # Test that the probability is respected.
    routing_table = RoutingTable(
        [
            RoutingRule(
                source_category="*",
                destination="provider1/model2",
                target=RoutingAction.REJECT,
                z_index=10,
                probability=0.5,
            )
        ]
    ).with_seed(0)

    reject_count = 0

    for _ in range(100):
        accept_map, rejected_models = routing_table.apply(("*",), {"model1-1", "model1-2", "model2", "model3"})
        reject_count += len(rejected_models)

    # With 50% probability, model2 should be rejected.
    assert reject_count / 100 == approx(0.5, rel=0.1)
