import numpy as np
from pytest import approx, mark

from ypl.backend.llm.reward import (
    REWARD_TIER_HIGH,
    REWARD_TIER_LOW,
    REWARD_TIER_MEDIUM,
    REWARD_TIER_VERY_LOW,
    UserTurnReward,
)


@mark.parametrize(
    "turn_quality_score, points, expected_tier_name",
    [
        (None, 100, REWARD_TIER_MEDIUM),
        (2, 100, REWARD_TIER_LOW),
        (5, 100, REWARD_TIER_MEDIUM),
        (9, 100, REWARD_TIER_HIGH),
        (None, 50000, REWARD_TIER_VERY_LOW),
        (5, 40000, REWARD_TIER_VERY_LOW),
    ],
)
def test_tiers(turn_quality_score: float | None, points: int, expected_tier_name: str) -> None:
    # Avoid using ctor to avoid DB calls in the __post_init__ method.
    user_turn_reward = UserTurnReward.__new__(UserTurnReward)
    user_turn_reward.turn_quality_score = turn_quality_score
    user_turn_reward.points = points

    tier = user_turn_reward.get_tier()
    assert tier.name == expected_tier_name

    expected_min, expected_max = tier.reward_range
    rewards_range = [user_turn_reward.get_tiered_reward(method="range") for _ in range(1000)]
    assert min(rewards_range) >= expected_min
    assert max(rewards_range) <= expected_max

    rewards_mean = [user_turn_reward.get_tiered_reward(method="mean") for _ in range(1000)]
    assert np.mean(rewards_mean) == approx(tier.mean_reward, rel=0.15)


@mark.parametrize(
    "is_new_user, is_inactive_user, is_first_turn, points, expected_probability",
    [
        (True, False, False, 100, 0.9),
        (False, True, False, 100, 0.9),
        (False, False, True, 100, 1.0),
        (False, False, False, 100, 0.8),
        (False, False, False, 5e5, 0.05),
    ],
)
def test_reward_probability(
    is_new_user: bool, is_inactive_user: bool, is_first_turn: bool, points: int, expected_probability: float
) -> None:
    # Avoid using ctor to avoid DB calls in the __post_init__ method.
    user_turn_reward = UserTurnReward.__new__(UserTurnReward)
    user_turn_reward.is_new_user = is_new_user
    user_turn_reward.is_inactive_user = is_inactive_user
    user_turn_reward.is_first_turn = is_first_turn
    user_turn_reward.points = points

    probabilities = [user_turn_reward.calculate_reward_probability() for _ in range(1000)]
    assert np.mean(probabilities) == approx(expected_probability, rel=0.15)
