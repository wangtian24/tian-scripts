import uuid
from collections.abc import Generator
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ChatMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pytest import approx, fixture, mark

import ypl.db.all_models  # noqa: F401
from ypl.backend.llm.reward import (
    FEEDBACK_REWARD_LOWER_BOUND,
    FEEDBACK_REWARD_UPPER_BOUND,
    QT_EVAL_REWARD_LOWER_BOUND,
    QT_EVAL_REWARD_UPPER_BOUND,
    UserTurnReward,
    _load_rules_constants,
    feedback_based_reward,
    qt_eval_reward,
    sign_up_reward,
)
from ypl.backend.tests.test_utils import MockSession
from ypl.db.rewards import RewardActionEnum, RewardAmountRule, RewardProbabilityRule

RULES = _load_rules_constants()

# Emulate what the DB returns: the rules from reward_rules.yml, reverse-sorted by priority.
MOCK_AMOUNT_RULES = sorted(
    [RewardAmountRule(**rule) for rule in RULES.get("amount_rules", [])], key=lambda x: -x.priority
)
MOCK_PROBABILITY_RULES = sorted(
    [
        RewardProbabilityRule(**rule)
        for rule in RULES.get("probability_rules", [])
        if rule.get("name") != "override_switch"
    ],
    key=lambda x: -x.priority,
)

# Fix the yaml parsing.
for rule in MOCK_AMOUNT_RULES + MOCK_PROBABILITY_RULES:
    action_type_str = rule.action_type
    if action_type_str and isinstance(action_type_str, str):
        rule.action_type = RewardActionEnum(action_type_str.lower())


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
    "turn_quality_score, points, points_last_month, expected_tier_name",
    [
        (None, 100, 100, "low"),
        (2, 100, 100, "low"),
        (5, 100, 100, "medium"),
        (9, 100, 100, "high"),
        (5, 100, 220000, "over_point_limit_last_month_low_reward"),
    ],
)
def test_tiers(turn_quality_score: float | None, points: int, expected_tier_name: str, points_last_month: int) -> None:
    user_turn_reward = create_user_turn_reward(
        turn_quality_score=turn_quality_score, points=points, points_last_month=points_last_month
    )
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
    "is_new_user, is_inactive_user, is_first_eval, points, points_last_day, expected_probability",
    [
        (True, False, False, 100, 100, 0.9),
        (False, True, False, 100, 100, 0.9),
        (False, False, True, 100, 100, 0.9),
        (False, False, False, 100, 100, 0.9),
        (False, False, False, 100, 100000, 0.0),
    ],
)
def test_reward_probability(
    is_new_user: bool,
    is_inactive_user: bool,
    is_first_eval: bool,
    points: int,
    expected_probability: float,
    points_last_day: int,
) -> None:
    user_turn_reward = create_user_turn_reward(
        is_new_user=is_new_user,
        is_inactive_user=is_inactive_user,
        is_first_eval=is_first_eval,
        points=points,
        points_last_day=points_last_day,
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
                ChatGeneration(message=ChatMessage(content='{"score": 3}', role="assistant"), generation_info=None)
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
                ChatGeneration(message=ChatMessage(content='{"score": 3}', role="assistant"), generation_info=None)
            ]
        )


@fixture(autouse=True)
def mock_chat_model() -> Generator[Any, None, None]:
    """Mock get_chat_model to prevent OpenAI client creation during import."""
    with patch("ypl.backend.llm.chat.get_chat_model") as mock:
        mock.return_value = MockLLM()
        yield mock


@patch("ypl.backend.llm.reward.get_async_engine")
@patch("ypl.backend.llm.reward.get_chat_model")
@patch("ypl.backend.llm.reward.GeminiLangChainAdapter")
@patch("ypl.backend.llm.reward._get_reward_points")
@patch("ypl.backend.llm.reward.Session")
@patch("ypl.backend.config.settings")
@patch("pydantic.PostgresDsn.build")
async def test_feedback_and_qt_eval_reward(
    mock_postgres_dsn: Any,
    mock_settings: Any,
    mock_session: Any,
    mock_get_reward_points: Any,
    mock_get_chat_model: Any,
    mock_gemini_adapter: Any,
    mock_engine: Any,
) -> None:
    # Mock PostgresDsn.build to return a valid URL string
    mock_postgres_dsn.return_value.unicode_string.return_value = "postgresql://test_user:test_pass@test_host/test_db"

    mock_session.return_value = MockSession()
    mock_engine.return_value = AsyncMock()

    # Configure both model mocks to return MockLLM
    mock_llm_instance = MockLLM()
    mock_get_chat_model.return_value = mock_llm_instance
    mock_gemini_adapter.return_value = mock_llm_instance

    def get_limits(daily: int, weekly: int, monthly: int) -> dict[timedelta, int]:
        return {
            timedelta(days=1): daily,
            timedelta(days=7): weekly,
            timedelta(days=30): monthly,
        }

    mock_get_reward_points.side_effect = lambda user_id, session, delta: get_limits(100, 500, 2000)[delta]

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
        assert rule_amount is not None
        assert rule_prob is not None

    # Test QT eval reward.
    for _ in range(10):
        qt_eval_result = await qt_eval_reward(test_user_id)
        assert qt_eval_result is not None
        should_reward, reward_amount, comment, rule_amount, rule_prob = qt_eval_result  # noqa
        assert should_reward is True
        assert rule_amount is not None
        assert rule_amount.name == "under_point_limits_qt_eval_reward"
        assert rule_prob is not None
        assert rule_prob.name == "under_point_limits_qt_eval_reward"
        assert QT_EVAL_REWARD_LOWER_BOUND <= reward_amount <= QT_EVAL_REWARD_UPPER_BOUND

    # Test no reward for high-point users.
    for args in (
        (60000, 0, 0),
        (0, 100005, 0),
        (0, 0, 200001),
    ):
        mock_get_reward_points.side_effect = lambda user_id, session, delta: get_limits(*args)[delta]  # noqa

        feedback_result = await feedback_based_reward(test_user_id, "")
        qt_eval_result = await qt_eval_reward(test_user_id)
        expected_rule_names = ["no_feedback_reward", "no_qt_eval_reward"]
        for result, expected_rule_name in zip([feedback_result, qt_eval_result], expected_rule_names, strict=True):
            should_reward, reward_amount, comment, rule_amount, rule_prob = result
            assert should_reward is False
            assert rule_prob is not None
            assert rule_prob.name == expected_rule_name


@patch("ypl.backend.llm.reward.get_async_engine")
@patch("ypl.backend.llm.reward.get_user_reward_count_by_action_type")
@patch("ypl.backend.llm.reward.Session")
@patch("ypl.backend.config.settings")
@patch("pydantic.PostgresDsn.build")
async def test_sign_up_reward(
    mock_postgres_dsn: Any,
    mock_settings: Any,
    mock_session: Any,
    mock_get_user_reward_count_by_action_type: Any,
    mock_engine: Any,
) -> None:
    # Mock PostgresDsn.build to return a valid URL string
    mock_postgres_dsn.return_value.unicode_string.return_value = "postgresql://test_user:test_pass@test_host/test_db"

    # Configure the mock session
    mock_session.return_value = MockSession()

    # Configure the mock engine to return a mock connection
    mock_engine_instance = AsyncMock()
    mock_engine.return_value = mock_engine_instance

    # Configure the user reward count mock
    mock_get_user_reward_count_by_action_type.side_effect = lambda user_id, action_type: 0

    test_user_id = "test_user"

    result = await sign_up_reward(test_user_id)
    should_reward, reward_amount, comment, rule_amount, rule_prob = result

    assert should_reward is True  # Should always reward sign up
    assert 1000 <= reward_amount <= 2000
    assert isinstance(comment, str)
    assert rule_amount is not None
    assert rule_amount.name == "base_sign_up_reward"
    assert rule_prob is not None
    assert rule_prob.name == "base_sign_up_reward"


@patch("ypl.backend.llm.reward.get_async_engine")
@patch("ypl.backend.llm.reward.get_user_reward_count_by_action_type")
@patch("ypl.backend.llm.reward.Session")
@patch("ypl.backend.config.settings")
@patch("pydantic.PostgresDsn.build")
async def test_sign_up_reward_no_repeat(
    mock_postgres_dsn: Any,
    mock_settings: Any,
    mock_session: Any,
    mock_get_user_reward_count_by_action_type: Any,
    mock_engine: Any,
) -> None:
    # Mock PostgresDsn.build to return a valid URL string
    mock_postgres_dsn.return_value.unicode_string.return_value = "postgresql://test_user:test_pass@test_host/test_db"

    mock_session.return_value = MockSession()
    mock_engine.return_value = AsyncMock()

    test_user_id = "test_user"

    mock_get_user_reward_count_by_action_type.side_effect = lambda user_id, action_type: 1

    result = await sign_up_reward(test_user_id)
    should_reward, reward_amount, comment, rule_amount, rule_prob = result
    assert should_reward is False  # should only reward once


@patch("ypl.backend.llm.reward.get_async_engine")
@patch("ypl.backend.llm.reward.get_chat_model")
@patch("ypl.backend.llm.reward.GeminiLangChainAdapter")
@patch("ypl.backend.llm.reward._get_reward_points")
@patch("ypl.backend.llm.reward.Session")
@patch("ypl.backend.config.settings")
@patch("pydantic.PostgresDsn.build")
def test_turn_reward_amount(
    mock_postgres_dsn: Any,
    mock_settings: Any,
    mock_session: Any,
    mock_get_reward_points: Any,
    mock_get_chat_model: Any,
    mock_gemini_adapter: Any,
    mock_engine: Any,
) -> None:
    daily_points_limit = RULES.get("constants", {}).get("daily_points_limit", None)
    assert daily_points_limit

    # Configure both model mocks to return MockLLM
    mock_llm_instance = MockLLM()
    mock_get_chat_model.return_value = mock_llm_instance
    mock_gemini_adapter.return_value = mock_llm_instance

    utr_low_points = create_user_turn_reward(
        points_last_day=10,
    )
    utr_mid_points = create_user_turn_reward(
        points_last_day=daily_points_limit * 0.6,
    )
    utr_high_points = create_user_turn_reward(
        points_last_day=daily_points_limit * 0.85,
    )
    amount_low_points = np.mean([utr_low_points.get_amount() for _ in range(20)])
    amount_mid_points = np.mean([utr_mid_points.get_amount() for _ in range(20)])
    amount_high_points = np.mean([utr_high_points.get_amount() for _ in range(20)])

    assert amount_low_points > amount_mid_points > amount_high_points
