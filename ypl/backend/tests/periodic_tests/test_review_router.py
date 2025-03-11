"""Tests for the review route classifier functionality."""

from typing import Any

import pytest

from ypl.backend.llm.review_router import get_review_type
from ypl.backend.llm.review_types import ReviewRoute

# Examples for testing the review route classifier
review_route_examples = [
    # PRO review examples (factual, mathematical, puzzle, or factoid problems)
    dict(
        content="What's the population of Tokyo in 2023?",
        expected_route=ReviewRoute.PRO,
    ),
    dict(
        content="Calculate the derivative of f(x) = 3x^2 + 2x - 5",
        expected_route=ReviewRoute.PRO,
    ),
    dict(
        content="Which programming language was developed first: Python or JavaScript?",
        expected_route=ReviewRoute.PRO,
    ),
    dict(
        content="Is there a bug in this SQL query: SELECT * FROM users WHERE user_id = 1; DROP TABLE users;",
        expected_route=ReviewRoute.PRO,
    ),
    dict(
        content="Who directed the movie Inception?",
        expected_route=ReviewRoute.PRO,
    ),
    dict(
        content="first man in space",
        expected_route=ReviewRoute.PRO,
    ),
    # CROSS review examples (elaborate, subjective queries)
    dict(
        content="Create a short story about a time traveler who can only go forward in time",
        expected_route=ReviewRoute.CROSS_CHECK,
    ),
    dict(
        content="What are the ethical implications of using AI in healthcare decision-making?",
        expected_route=ReviewRoute.CROSS_CHECK,
    ),
    dict(
        content="How might cities evolve over the next 50 years to address climate change?",
        expected_route=ReviewRoute.CROSS_CHECK,
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("example", review_route_examples)
async def test_review_router_direct(example: dict[str, Any]) -> None:
    """Test the get_review_type function directly."""
    result = await get_review_type(example["content"])
    # Check that the result matches the expected route
    assert (
        result == example["expected_route"]
    ), f"Expected {example['expected_route']}, got {result} for '{example['content']}'"
