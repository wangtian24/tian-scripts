import uuid
from collections.abc import Generator
from datetime import datetime
from typing import Any
from unittest.mock import patch

import numpy as np
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ChatMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
)
from pytest import approx, fixture, mark

import ypl.db.all_models  # noqa: F401
from ypl.backend.llm.reward import (
    FEEDBACK_REWARD_LOWER_BOUND,
    FEEDBACK_REWARD_UPPER_BOUND,
    MEAN_EVAL_REWARD,
    REWARD_TIER_HIGH,
    REWARD_TIER_LOW,
    REWARD_TIER_MEDIUM,
    REWARD_TIER_VERY_LOW,
    UserTurnReward,
    feedback_based_reward,
)
from ypl.db.rewards import RewardActionEnum, RewardAmountRule, RewardProbabilityRule

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
        action_type=RewardActionEnum.TURN,
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
        action_type=RewardActionEnum.TURN,
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
        action_type=RewardActionEnum.TURN,
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
        action_type=RewardActionEnum.TURN,
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
        action_type=RewardActionEnum.TURN,
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
        name="first_eval",
        priority=250,
        action_type=RewardActionEnum.TURN,
        conditions={
            "all": [
                {
                    "name": "is_first_eval",
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
        action_type=RewardActionEnum.TURN,
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
        action_type=RewardActionEnum.TURN,
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
    user_turn_reward.action_type = RewardActionEnum.TURN
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
    "is_new_user, is_inactive_user, is_first_eval, credits, expected_probability",
    [
        (True, False, False, 100, 0.9),
        (False, True, False, 100, 0.9),
        (False, False, True, 100, 1.0),
        (False, False, False, 100, 0.8),
        (False, False, False, 5e5, 0.05),
    ],
)
def test_reward_probability(
    is_new_user: bool, is_inactive_user: bool, is_first_eval: bool, credits: int, expected_probability: float
) -> None:
    user_turn_reward = create_user_turn_reward(
        is_new_user=is_new_user, is_inactive_user=is_inactive_user, is_first_eval=is_first_eval, points=credits
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


# Create a mock LLM for testing
class MockLLM(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(message=ChatMessage(content='{"score": 7}', role="assistant"), generation_info=None)
            ]
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(message=ChatMessage(content='{"score": 7}', role="assistant"), generation_info=None)
            ]
        )


@fixture(autouse=True)
def mock_chat_model() -> Generator[Any, None, None]:
    """Mock get_chat_model to prevent OpenAI client creation during import."""
    with patch("ypl.backend.llm.chat.get_chat_model") as mock:
        mock.return_value = MockLLM()
        yield mock


@patch("ypl.backend.llm.reward.get_reward_llm")
async def test_feedback_reward(mock_get_llm: Any) -> None:
    # Setup mock chain
    mock_llm = MockLLM()
    mock_get_llm.return_value = mock_llm

    test_user_id = "test_user"
    # Test cases
    test_cases = [
        ("Great feedback", True),  # Good feedback
        ("ok", True),  # Short feedback
        ("", True),  # Empty feedback
    ]

    for feedback, should_reward in test_cases:
        result = await feedback_based_reward(test_user_id, feedback)
        should_reward, reward_amount, comment, rule_amount, rule_prob = result

        assert should_reward is True  # Should always reward feedback
        assert FEEDBACK_REWARD_LOWER_BOUND <= reward_amount <= FEEDBACK_REWARD_UPPER_BOUND
        assert isinstance(comment, str)
        assert rule_amount is None
        assert rule_prob is None
