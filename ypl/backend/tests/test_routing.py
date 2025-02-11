import random
import uuid
from collections import Counter
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from pytest import approx, mark

from ypl.backend.config import settings
from ypl.backend.llm.chat import SelectIntent, SelectModelsV2Request, select_models_plus
from ypl.backend.llm.prompt_selector import CategorizedPromptModifierSelector
from ypl.backend.llm.ranking import Battle, ChoixRanker, ChoixRankerConfIntervals, EloRanker
from ypl.backend.llm.routing.modules.filters import ContextLengthFilter, SupportsImageAttachmentModelFilter, TopK
from ypl.backend.llm.routing.modules.proposers import (
    AlwaysGoodModelMetaRouter,
    ConfidenceIntervalWidthModelProposer,
    EloProposer,
    MinimumFractionModelProposer,
    ProportionalModelProposer,
    RandomModelProposer,
    _fast_compute_all_conf_overlap_diffs,
    _fast_compute_all_num_intersections,
)
from ypl.backend.llm.routing.policy import SelectionCriteria, exponential_decay
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.routing.router import (
    get_simple_pro_router,
)
from ypl.backend.llm.routing.router_state import RouterState
from ypl.backend.llm.routing.rule_router import RoutingTable
from ypl.backend.tests.utils import get_battles
from ypl.db.language_models import RoutingAction, RoutingRule

settings.ROUTING_DO_LOGGING = False
ACTIVE_MODELS = [
    "Qwen/Qwen2-72B-Instruct",
    "ai21/jamba-1-5-large",
    "ai21/jamba-1-5-mini",
    "amazon/nova-pro-v1",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "codestral-2405",
    "cohere/command-r",
    "cohere/command-r-08-2024",
    "cohere/command-r-plus",
    "cohere/command-r-plus-08-2024",
    "databricks/dbrx-instruct",
    "deepseek-coder",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-r1",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro-002",
    "gemini-1.5-pro-002-online",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-thinking-exp-1219",
    "gemini-exp-1206",
    "gemma2-9b-it",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gryphe/mythomax-l2-13b",
    "llama-3.1-8b-instant",
    "llama-3.1-sonar-huge-128k-online",
    "llama-3.1-sonar-large-128k-chat",
    "llama-3.1-sonar-large-128k-online",
    "llama-3.1-sonar-small-128k-chat",
    "llama-3.1-sonar-small-128k-online",
    "llama-3.3-70b",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama3.1-70b",
    "llama3.1-8b",
    "meta-llama/Llama-3-70b-chat-hf",
    "meta-llama/Llama-3-8b-chat-hf",
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.2-1b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "microsoft/phi-3-medium-128k-instruct",
    "microsoft/phi-3.5-mini-128k-instruct",
    "microsoft/phi-4",
    "ministral-3b-2410",
    "ministral-8b-2410",
    "mistral-large-2402",
    "mistral-medium",
    "mixtral-8x7b-32768",
    "models/gemini-1.5-pro-002",
    "models/gemini-2.0-flash-exp",
    "nousresearch/hermes-3-llama-3.1-70b",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "o1-mini-2024-09-12",
    "o1-preview-2024-09-12",
    "open-mistral-nemo-2407",
    "open-mixtral-8x22b",
    "open-mixtral-8x7b",
    "openai/o1",
    "pixtral-12b-2409",
    "qwen-max",
    "qwen-plus",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct",
    "x-ai/grok-2-1212",
    "x-ai/grok-beta",
]
IMAGE_ATTACHMENT_MODELS = [
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro-002",
    "gemini-2.0-flash-exp",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "pixtral-12b-2409",
    "x-ai/grok-2-1212",
    "x-ai/grok-beta",
]
PRO_MODELS = ["pro1", "pro2", "pro3", "pro4"]
STRONG_MODELS = [
    "cohere/command-r-plus",
    "deepseek-coder",
    "gemini-1.5-flash-002",
    "gemini-2.0-flash-exp",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "llama-3.3-70b",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama3.1-70b",
    "llama3.1-8b",
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "mixtral-8x7b-32768",
    "o1-mini-2024-09-12",
    "x-ai/grok-2-1212",
    "x-ai/grok-beta",
]
PROVIDER_MAP = {
    "Qwen/Qwen2-72B-Instruct": "alibaba",
    "ai21/jamba-1-5-large": "ai21",
    "ai21/jamba-1-5-mini": "ai21",
    "amazon/nova-pro-v1": "amazon",
    "claude-3-5-haiku-20241022": "anthropic",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-haiku-20240307": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-sonnet-20240229": "anthropic",
    "codestral-2405": "mistralai",
    "cohere/command-r": "cohere",
    "cohere/command-r-08-2024": "cohere",
    "cohere/command-r-plus": "cohere",
    "cohere/command-r-plus-08-2024": "cohere",
    "databricks/dbrx-instruct": "databricks",
    "deepseek-coder": "deepseek",
    "deepseek/deepseek-chat": "deepseek",
    "deepseek/deepseek-r1": "deepseek",
    "gemini-1.5-flash-002": "google",
    "gemini-1.5-flash-8b": "google",
    "gemini-1.5-pro-002": "google",
    "gemini-1.5-pro-002-online": "google",
    "gemini-2.0-flash-exp": "google",
    "gemini-2.0-flash-thinking-exp-1219": "google",
    "gemini-exp-1206": "google",
    "gemma2-9b-it": "google",
    "google/gemma-2-27b-it": "google",
    "google/gemma-2-9b-it": "google",
    "gpt-3.5-turbo-0125": "openai",
    "gpt-4-turbo": "openai",
    "gpt-4o-2024-05-13": "openai",
    "gpt-4o-2024-08-06": "openai",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4o-mini-2024-07-18": "openai",
    "gryphe/mythomax-l2-13b": "gryphe",
    "llama-3.1-8b-instant": "meta",
    "llama-3.1-sonar-huge-128k-online": "meta",
    "llama-3.1-sonar-large-128k-chat": "meta",
    "llama-3.1-sonar-large-128k-online": "meta",
    "llama-3.1-sonar-small-128k-chat": "meta",
    "llama-3.1-sonar-small-128k-online": "meta",
    "llama-3.3-70b": "meta",
    "llama-3.3-70b-versatile": "meta",
    "llama3-70b-8192": "meta",
    "llama3-8b-8192": "meta",
    "llama3.1-70b": "meta",
    "llama3.1-8b": "meta",
    "meta-llama/Llama-3-70b-chat-hf": "meta",
    "meta-llama/Llama-3-8b-chat-hf": "meta",
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": "meta",
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": "meta",
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": "meta",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "meta",
    "meta-llama/llama-3.1-405b-instruct": "meta",
    "meta-llama/llama-3.1-70b-instruct": "meta",
    "meta-llama/llama-3.1-8b-instruct": "meta",
    "meta-llama/llama-3.2-1b-instruct": "meta",
    "meta-llama/llama-3.3-70b-instruct": "meta",
    "microsoft/phi-3-medium-128k-instruct": "azure",
    "microsoft/phi-3.5-mini-128k-instruct": "azure",
    "microsoft/phi-4": "azure",
    "ministral-3b-2410": "mistralai",
    "ministral-8b-2410": "mistralai",
    "mistral-large-2402": "mistralai",
    "mistral-medium": "mistralai",
    "mixtral-8x7b-32768": "mistralai",
    "models/gemini-1.5-pro-002": "google",
    "models/gemini-2.0-flash-exp": "google",
    "nousresearch/hermes-3-llama-3.1-70b": "nousresearch",
    "nvidia/llama-3.1-nemotron-70b-instruct": "nvidia",
    "o1-mini-2024-09-12": "openai",
    "o1-preview-2024-09-12": "openai",
    "open-mistral-nemo-2407": "mistralai",
    "open-mixtral-8x22b": "mistralai",
    "open-mixtral-8x7b": "mistralai",
    "openai/o1": "openai",
    "pixtral-12b-2409": "mistralai",
    "qwen-max": "alibaba",
    "qwen-plus": "alibaba",
    "qwen/qwen-2.5-72b-instruct": "alibaba",
    "qwen/qwen-2.5-coder-32b-instruct": "alibaba",
    "x-ai/grok-2-1212": "x-ai",
    "x-ai/grok-beta": "x-ai",
}


async def mock_online_yupp_fn(*args: Any, **kwargs: Any) -> str:
    return "advice"


async def mock_topic_categorizer_fn(*args: Any, **kwargs: Any) -> list[str]:
    return []


async def mock_modifier_labeler_fn(*args: Any, **kwargs: Any) -> list[str]:
    return ["concise"]


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


# These patches must be in the resverse order of the argument list below,
# and the path is where they are called rather than where they are defined.
@patch("ypl.backend.llm.promotions.get_model_creation_dates", return_value={})
@patch("ypl.backend.llm.promotions.get_active_model_promotions", return_value=[])
@patch("ypl.backend.llm.routing.router.HighErrorRateFilter.select_models")
@patch("ypl.backend.llm.routing.modules.proposers.get_all_strong_models")
@patch("ypl.backend.llm.routing.modules.proposers.get_all_pro_models")
@patch("ypl.backend.llm.routing.modules.rankers.deduce_model_speed_scores")
@patch("ypl.backend.llm.routing.modules.proposers.deduce_original_providers")
@patch("ypl.backend.llm.routing.modules.filters.deduce_original_providers")
@patch("ypl.backend.llm.routing.rule_router.deduce_original_providers")
@patch("ypl.backend.llm.routing.modules.rankers.deduce_semantic_groups")
@patch("ypl.backend.llm.routing.modules.filters.deduce_semantic_groups")
@patch("ypl.backend.llm.routing.rule_router.get_routing_table")
@patch("ypl.backend.llm.routing.modules.filters.get_image_attachment_models")
@patch("ypl.backend.llm.routing.modules.filters.get_active_models")
@patch("ypl.backend.llm.routing.modules.filters.get_model_context_lengths")
@patch("ypl.backend.llm.chat.has_image_attachments")
@pytest.mark.asyncio
async def test_simple_pro_router(
    mock_has_image_attachments: Mock,
    mock_model_context_lengths: Mock,
    mock_active_models: Mock,
    mock_image_attachment_models: Mock,
    mock_routing_table: Mock,
    mock_semantic_group_map1: Mock,
    mock_semantic_group_map2: Mock,
    mock_deduce_providers1: Mock,
    mock_deduce_providers2: Mock,
    mock_deduce_providers3: Mock,
    mock_deduce_speed_scores: Mock,
    mock_get_all_pro_models: Mock,
    mock_get_all_strong_models: Mock,
    mock_error_filter: Mock,
    mock_get_active_model_promotions: Mock,
    mock_get_model_creation_dates: Mock,
) -> None:
    mock_table = Mock()
    mock_table.apply.return_value = ({}, set())  # empty accept_map and rejected_models
    mock_routing_table.return_value = mock_table

    # Test that we get different models
    pro_models = {"pro1", "pro2", "pro3", "pro4"}
    mock_get_all_pro_models.return_value = pro_models
    mock_semantic_group_map1.return_value = {}
    mock_semantic_group_map2.return_value = {}
    mock_has_image_attachments.return_value = False
    models = {"model1", "model2"}
    reputable_providers = {"pro1", "pro2", "pro3", "model1"}
    all_models = models | pro_models
    mock_active_models.return_value = all_models
    mock_image_attachment_models.return_value = all_models
    state = RouterState(all_models=all_models)
    # Just make a provider for each model named after the model.
    mock_deduce_providers1.return_value = {model: model for model in all_models}
    mock_deduce_providers2.return_value = {model: model for model in all_models}
    mock_deduce_providers3.return_value = {model: model for model in all_models}
    mock_deduce_speed_scores.return_value = {model: 1.0 for model in all_models}
    mock_get_all_strong_models.return_value = {"pro1", "pro2", "pro3", "model1"}
    mock_error_filter.side_effect = lambda state: state

    mock_model_context_lengths.return_value = {model: 1000000 for model in all_models}

    all_selected_models = set()
    for _ in range(50):
        router = await get_simple_pro_router(
            prompt="",
            num_models=2,
            reputable_providers=reputable_providers,
            preference=RoutingPreference(turns=[], user_id="123", user_selected_models=[], same_turn_shown_models=[]),
        )
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
            prompt="",
            num_models=2,
            reputable_providers=reputable_providers,
            preference=RoutingPreference(
                turns=[], user_id="123", user_selected_models=["model1", "model2"], same_turn_shown_models=[]
            ),
            required_models=["model1", "model2"],
        )
        selected_models = router.select_models(state=state).get_sorted_selected_models()
        assert selected_models == ["model1", "model2"]

    for _ in range(10):
        router = await get_simple_pro_router(
            prompt="",
            num_models=1,
            reputable_providers=reputable_providers,
            preference=RoutingPreference(
                turns=[], user_id="123", user_selected_models=["model1"], same_turn_shown_models=[]
            ),
            required_models=["model1"],
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
    mock_deduce_providers3.return_value = {model: model for model in all_models}
    mock_deduce_speed_scores.return_value = {model: 1.0 for model in all_models}
    mock_get_all_strong_models.return_value = {"pro1", "pro2", "pro3", "model1"}
    mock_error_filter.side_effect = lambda state: state
    mock_semantic_group_map1.return_value = {"model1": "group1", "model2": "group1"}
    mock_semantic_group_map2.return_value = {"model1": "group1", "model2": "group1"}

    router = await get_simple_pro_router(
        prompt="",
        num_models=1,
        reputable_providers=reputable_providers,
        preference=RoutingPreference(
            turns=[], user_id="123", user_selected_models=["model2"], same_turn_shown_models=[]
        ),
        required_models=["model2"],
    )

    for _ in range(15):
        state = RouterState(all_models=models)
        selected_models = router.select_models(state=state).get_sorted_selected_models()
        assert "model2" in selected_models and "model1" not in selected_models

    router_two_selected_models = await get_simple_pro_router(
        prompt="",
        num_models=2,
        reputable_providers=reputable_providers,
        preference=RoutingPreference(
            turns=[], user_id="123", user_selected_models=["model2", "model1"], same_turn_shown_models=[]
        ),
        required_models=["model2", "model1"],
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


@patch("ypl.backend.llm.routing.modules.filters.get_image_attachment_models")
@patch("ypl.backend.llm.routing.modules.filters.get_active_models")
def test_supports_image_attachment_filter(mock_active_models: Mock, mock_image_attachment_models: Mock) -> None:
    all_models = {"m1", "m2", "m3", "m4"}
    image_models = {"m1", "m2"}
    mock_active_models.return_value = all_models
    mock_image_attachment_models.return_value = image_models
    c = {SelectionCriteria.RANDOM: 1.0}
    state = RouterState(all_models=all_models, selected_models={m: c for m in all_models})
    filter = SupportsImageAttachmentModelFilter()

    state, rejected_models = filter._filter(state)
    assert state.excluded_models == {"m3", "m4"}
    assert rejected_models == {"m3", "m4"}
    assert state.selected_models == {"m1": c, "m2": c}


@patch("ypl.backend.llm.routing.modules.filters.get_model_context_lengths")
def test_context_length_filter(mock_context_lengths: Mock) -> None:
    context_lengths = {"m1": 1000, "m2": 1000, "m3": 100}
    all_models = set(context_lengths.keys())
    mock_context_lengths.return_value = context_lengths
    c = {SelectionCriteria.RANDOM: 1.0}

    def check_excluded_models_by_prompt_len(prompt_len_tokens: int, expected_excluded_models: set[str]) -> None:
        state = RouterState(all_models=all_models, selected_models={m: c for m in all_models})
        filter = ContextLengthFilter(prompt="hi " * prompt_len_tokens, max_length_fraction=0.5)
        state, rejected_models = filter._filter(state)
        assert state.excluded_models == rejected_models == expected_excluded_models
        assert state.selected_models == {m: c for m in all_models - expected_excluded_models}

    check_excluded_models_by_prompt_len(600, {"m3"})
    check_excluded_models_by_prompt_len(60, set())


@pytest.mark.asyncio
@patch("ypl.backend.llm.prompt_modifier.PromptModifierLabeler")
@patch("ypl.backend.llm.category_labeler.YuppMultilabelClassifier")
@patch("ypl.backend.llm.category_labeler.YuppOnlinePromptLabeler")
@patch("ypl.backend.llm.chat.get_prompt_categories", return_value=[])
@patch("ypl.backend.llm.chat.get_prompt_modifiers", return_value=[])
@patch("ypl.backend.llm.chat.label_turn_quality", return_value=None)
@patch("ypl.backend.llm.chat.get_chat_required_models", return_value=[])
@patch("ypl.backend.llm.routing.modules.filters.get_active_models", return_value=ACTIVE_MODELS)
@patch("ypl.backend.llm.routing.modules.filters.get_image_attachment_models", return_value=IMAGE_ATTACHMENT_MODELS)
@patch("ypl.backend.llm.routing.modules.proposers.get_image_attachment_models", return_value=IMAGE_ATTACHMENT_MODELS)
@patch("ypl.backend.llm.routing.modules.proposers.get_all_pro_models", return_value=PRO_MODELS)
@patch("ypl.backend.llm.routing.modules.proposers.get_all_strong_models", return_value=STRONG_MODELS)
@patch("ypl.backend.llm.chat.get_preferences")
@patch("ypl.backend.llm.routing.rule_router.get_routing_table", return_value=RoutingTable([]))
@patch(
    "ypl.backend.llm.routing.modules.filters.get_model_context_lengths",
    return_value={m: 1000000 for m in ACTIVE_MODELS},
)
@patch("ypl.backend.llm.chat.store_modifiers", return_value=None)
@patch("ypl.backend.llm.routing.router._get_good_and_bad_models", return_value=(set(), set()))
@patch("ypl.backend.llm.routing.router_state.RouterState.get_all_models", return_value=set(ACTIVE_MODELS))
@patch("ypl.backend.llm.promotions.get_model_creation_dates", return_value={})
@patch("ypl.backend.llm.promotions.get_active_model_promotions", return_value=[])
@patch("ypl.backend.llm.routing.modules.rankers.deduce_model_speed_scores", return_value={})
@patch("ypl.backend.llm.routing.modules.rankers.deduce_semantic_groups", return_value={})
@patch("ypl.backend.llm.routing.modules.filters.deduce_semantic_groups", return_value={})
@patch("ypl.backend.llm.routing.modules.filters.deduce_original_providers", return_value=PROVIDER_MAP)
@patch("ypl.backend.llm.routing.modules.proposers.deduce_original_providers", return_value=PROVIDER_MAP)
@patch("ypl.backend.llm.routing.rule_router.deduce_original_providers", return_value=PROVIDER_MAP)
@patch("ypl.backend.llm.chat.deduce_original_providers", return_value=PROVIDER_MAP)
@patch("ypl.backend.llm.routing.modules.filters.HighErrorRateFilter._get_error_rates", return_value={})
@patch("ypl.backend.llm.prompt_selector.CategorizedPromptModifierSelector.make_default_from_db")
@patch("ypl.backend.llm.chat.get_user_message", return_value="hi")
@patch("ypl.backend.llm.chat.get_modifiers_by_model_and_position", return_value=({}, (None, None)))
async def test_select_models_plus(
    mock_get_modifiers_by_model_and_position: AsyncMock,
    mock_get_user_message: Mock,
    mock_make_default_from_db: Mock,
    mock_high_error_rate_filter: Mock,
    mock_deduce_original_providers1: Mock,
    mock_deduce_original_providers2: Mock,
    mock_deduce_original_providers3: Mock,
    mock_deduce_original_providers4: Mock,
    mock_deduce_semantic_groups1: Mock,
    mock_deduce_semantic_groups2: Mock,
    mock_deduce_model_speed_scores: Mock,
    mock_get_active_model_promotions: Mock,
    mock_get_model_creation_dates: Mock,
    mock_get_all_models: Mock,
    mock_get_good_and_bad_models: Mock,
    mock_store_modifiers: Mock,
    mock_get_model_context_lengths: Mock,
    mock_get_routing_table: Mock,
    mock_get_preferences: Mock,
    mock_get_all_strong_models: Mock,
    mock_get_all_pro_models: Mock,
    mock_get_image_attachment_models1: Mock,
    mock_get_image_attachment_models2: Mock,
    mock_active_models: Mock,
    mock_get_chat_required_models: Mock,
    mock_label_turn_quality: Mock,
    mock_get_prompt_modifiers: Mock,
    mock_get_prompt_categories: Mock,
    MockOnlineYupp: Mock,
    MockTopicCategorizer: Mock,
    MockModifierLabeler: Mock,
) -> None:
    mock_get_preferences.side_effect = lambda *args, **kwargs: RoutingPreference(
        turns=[], user_id="user", user_selected_models=[], same_turn_shown_models=[]
    )
    mock_make_default_from_db.return_value = CategorizedPromptModifierSelector.make_default()
    show_me_more_models = random.sample(IMAGE_ATTACHMENT_MODELS, 3)

    def make_image_request(intent: SelectIntent) -> SelectModelsV2Request:
        if intent == SelectIntent.SHOW_ME_MORE:
            mock_get_preferences.side_effect = lambda *args, **kwargs: RoutingPreference(
                turns=[],
                user_id="user",
                user_selected_models=[],
                same_turn_shown_models=show_me_more_models,
            )
        else:
            mock_get_preferences.side_effect = lambda *args, **kwargs: RoutingPreference(
                turns=[], user_id="user", user_selected_models=[], same_turn_shown_models=[]
            )

        return SelectModelsV2Request(
            prompt=None if intent == SelectIntent.SHOW_ME_MORE else "hi",
            num_models=2,
            intent=intent,
            chat_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            provided_categories=["image"],
            debug_level=0,
            prompt_modifier_id=None,
            user_id="user",
        )

    MockOnlineYupp.return_value.alabel = mock_online_yupp_fn
    MockTopicCategorizer.return_value.alabel = mock_topic_categorizer_fn
    MockModifierLabeler.return_value.alabel = mock_modifier_labeler_fn

    for _ in range(30):
        for intent in (SelectIntent.NEW_TURN, SelectIntent.SHOW_ME_MORE, SelectIntent.NEW_CHAT):
            request = make_image_request(intent)
            response = await select_models_plus(request)
            assert len(response.models) == 2
            assert len(response.fallback_models) == 2
            assert all(m[0] in IMAGE_ATTACHMENT_MODELS for m in response.models)
            assert all(m[0] in IMAGE_ATTACHMENT_MODELS for m in response.fallback_models)
