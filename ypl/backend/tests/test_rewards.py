import uuid
from datetime import datetime
from typing import Any
from unittest.mock import patch

import numpy as np
from pytest import approx, fixture, mark

from ypl.backend.llm.reward import (
    MEAN_EVAL_REWARD,
    REWARD_TIER_HIGH,
    REWARD_TIER_LOW,
    REWARD_TIER_MEDIUM,
    REWARD_TIER_VERY_LOW,
    UserTurnReward,
)
from ypl.db.rewards import RewardAmountRule, RewardProbabilityRule

MOCK_AMOUNT_RULES = [
    RewardAmountRule(
        reward_amount_rule_id=uuid.uuid4(),
        name=REWARD_TIER_VERY_LOW,
        priority=200,
        conditions={
            "all": [
                {
                    "name": "credits",
                    "operator": "greater_than_or_equal_to",
                    "value": 20000,
                }
            ]
        },
        min_value=0,
        max_value=10,
        mean_value=0,
        comments=["Thank you for participating."],
    ),
    RewardAmountRule(
        reward_amount_rule_id=uuid.uuid4(),
        name=REWARD_TIER_HIGH,
        priority=100,
        conditions={
            "all": [
                {
                    "name": "turn_quality_score",
                    "operator": "greater_than_or_equal_to",
                    "value": 8,
                }
            ]
        },
        min_value=200,
        max_value=1000,
        mean_value=MEAN_EVAL_REWARD * 1.5,
        comments=[
            "This engages the model in a well-rounded, thought-provoking way.",
            "This prompt is a great conversation starter.",
            "This prompt challenges the model effectively across many aspects.",
        ],
    ),
    RewardAmountRule(
        reward_amount_rule_id=uuid.uuid4(),
        name=REWARD_TIER_MEDIUM,
        is_default=True,
        priority=90,
        conditions={
            "all": [
                {
                    "name": "turn_quality_score",
                    "operator": "greater_than_or_equal_to",
                    "value": 4,
                }
            ]
        },
        min_value=50,
        max_value=200,
        mean_value=MEAN_EVAL_REWARD * 1.0,
        comments=[
            "Try writing a more novel prompt for better rewards.",
            "Try adding more complexity to future prompts to earn more rewards.",
            "Try more differentiated prompts for higher reward.",
        ],
    ),
    RewardAmountRule(
        reward_amount_rule_id=uuid.uuid4(),
        name=REWARD_TIER_LOW,
        priority=80,
        conditions={
            "all": [
                {
                    "name": "turn_quality_score",
                    "operator": "greater_than_or_equal_to",
                    "value": 1,
                }
            ]
        },
        min_value=10,
        max_value=50,
        mean_value=MEAN_EVAL_REWARD * 0.5,
        comments=["Thank you for participating."],
    ),
]

MOCK_PROBABILITY_RULES = [
    RewardProbabilityRule(
        reward_probability_rule_id=uuid.uuid4(),
        name="high_credits",
        priority=300,
        conditions={
            "all": [
                {
                    "name": "credits",
                    "operator": "greater_than_or_equal_to",
                    "value": 20000,
                }
            ]
        },
        probability=0.05,
    ),
    RewardProbabilityRule(
        reward_probability_rule_id=uuid.uuid4(),
        name="first_turn",
        priority=250,
        conditions={
            "all": [
                {
                    "name": "is_first_turn",
                    "operator": "is_true",
                    "value": True,
                }
            ]
        },
        probability=1.0,
    ),
    RewardProbabilityRule(
        reward_probability_rule_id=uuid.uuid4(),
        name="new_or_inactive_user",
        priority=150,
        conditions={
            "any": [
                {
                    "name": "is_new_user",
                    "operator": "is_true",
                    "value": True,
                },
                {
                    "name": "is_inactive_user",
                    "operator": "is_true",
                    "value": True,
                },
            ]
        },
        probability=0.9,
    ),
    RewardProbabilityRule(
        reward_probability_rule_id=uuid.uuid4(),
        name="active_user",
        priority=0,
        is_default=True,
        conditions={},  # Always matches.
        probability=0.8,
    ),
]


@fixture(autouse=True)
def mock_rules() -> Any:
    with (
        patch("ypl.backend.llm.reward.get_reward_amount_rules", return_value=MOCK_AMOUNT_RULES),
        patch("ypl.backend.llm.reward.get_reward_probability_rules", return_value=MOCK_PROBABILITY_RULES),
    ):
        yield


def create_user_turn_reward(**kwargs: Any) -> UserTurnReward:
    # Avoid using ctor to avoid DB calls in the __post_init__ method.
    user_turn_reward = UserTurnReward.__new__(UserTurnReward)
    user_turn_reward.user_id = "fake_user_id"
    user_turn_reward.turn_id = uuid.uuid4()
    for key, value in kwargs.items():
        if value is not None:
            setattr(user_turn_reward, key, value)
    user_turn_reward.amount_rule = user_turn_reward._get_amount_rule()
    user_turn_reward.probability_rule = user_turn_reward._get_probability_rule()
    return user_turn_reward


@mark.parametrize(
    "turn_quality_score, credits, expected_tier_name",
    [
        (None, 100, REWARD_TIER_MEDIUM),
        (2, 100, REWARD_TIER_LOW),
        (5, 100, REWARD_TIER_MEDIUM),
        (9, 100, REWARD_TIER_HIGH),
        (None, 50000, REWARD_TIER_VERY_LOW),
        (5, 40000, REWARD_TIER_VERY_LOW),
    ],
)
def test_tiers(turn_quality_score: float | None, credits: int, expected_tier_name: str) -> None:
    user_turn_reward = create_user_turn_reward(turn_quality_score=turn_quality_score, points=credits)

    rule = user_turn_reward._get_amount_rule()
    assert rule is not None
    assert rule.name == expected_tier_name

    expected_min, expected_max = rule.min_value, rule.max_value
    rewards_range = [user_turn_reward.get_amount(method="range") for _ in range(1000)]
    assert min(rewards_range) >= expected_min
    assert max(rewards_range) <= expected_max

    rewards_mean = [user_turn_reward.get_amount(method="mean") for _ in range(1000)]
    assert np.mean(rewards_mean) == approx(rule.mean_value, rel=0.15)


@mark.parametrize(
    "is_new_user, is_inactive_user, is_first_turn, credits, expected_probability",
    [
        (True, False, False, 100, 0.9),
        (False, True, False, 100, 0.9),
        (False, False, True, 100, 1.0),
        (False, False, False, 100, 0.8),
        (False, False, False, 5e5, 0.05),
    ],
)
def test_reward_probability(
    is_new_user: bool, is_inactive_user: bool, is_first_turn: bool, credits: int, expected_probability: float
) -> None:
    user_turn_reward = create_user_turn_reward(
        is_new_user=is_new_user, is_inactive_user=is_inactive_user, is_first_turn=is_first_turn, points=credits
    )

    probabilities = [user_turn_reward.get_probability() for _ in range(1000)]
    assert np.mean(probabilities) == approx(expected_probability, rel=0.05)


def test_reward_rule_equality() -> None:
    assert MOCK_AMOUNT_RULES[0] == MOCK_AMOUNT_RULES[0]
    assert MOCK_PROBABILITY_RULES[0] == MOCK_PROBABILITY_RULES[0]
    assert MOCK_AMOUNT_RULES[0] != MOCK_AMOUNT_RULES[1]
    assert MOCK_PROBABILITY_RULES[0] != MOCK_PROBABILITY_RULES[1]

    # Should match even if IDs/timestamps differ.
    assert MOCK_AMOUNT_RULES[0] == MOCK_AMOUNT_RULES[0].model_copy(update={"reward_amount_rule_id": uuid.uuid4()})
    assert MOCK_PROBABILITY_RULES[0] == MOCK_PROBABILITY_RULES[0].model_copy(update={"created_at": datetime.now()})
