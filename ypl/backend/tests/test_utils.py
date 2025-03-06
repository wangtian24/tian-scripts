import time
from collections.abc import Iterable
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest import approx, raises

from ypl.backend.llm.utils import ThresholdCounter, norm_softmax
from ypl.utils import Delegator, EarlyTerminatedException, extract_json_dict_from_text


def check_array_approx(ar1: Iterable[float], ar2: Iterable[float]) -> None:
    for a, b in zip(ar1, ar2, strict=True):
        assert a == approx(b)


def test_norm_softmax() -> None:
    check_array_approx(norm_softmax([0, 0, 0]), [0.33333333, 0.33333333, 0.33333333])
    check_array_approx(norm_softmax([-2, -1, 0]), [0.29632715, 0.33032048, 0.37335237])
    check_array_approx(norm_softmax([1, 2, 3]), [0.30850836, 0.33359748, 0.35789417])
    check_array_approx(norm_softmax([-1, 0, 1]), [0.25991820, 0.32747953, 0.41260228])
    check_array_approx(norm_softmax([1, 1]), [0.5, 0.5])
    check_array_approx(norm_softmax([]), [])
    check_array_approx(norm_softmax([0]), [1])

    for n in [None, np.inf, -np.inf, np.nan]:
        with raises(ValueError):
            norm_softmax([n, 1])  # type: ignore


def test_update_ratings_counter() -> None:
    counter = ThresholdCounter(threshold=3, max_threshold=10, growth_rate=2)

    assert counter.count == 0
    assert counter.total_count == 0
    assert counter.threshold == 3

    for _ in range(2):
        counter.increment()
        assert not counter.is_threshold_reached()

    counter.increment()
    assert counter.is_threshold_reached()
    assert counter.count == 3
    assert counter.total_count == 3

    counter.reset()
    assert counter.count == 0
    assert counter.total_count == 3
    assert counter.threshold == 6
    assert not counter.is_threshold_reached()

    # Test multiple cycles
    for _ in range(25):
        counter.increment()
        if counter.is_threshold_reached():
            counter.reset()
        assert counter.count < counter.threshold

    assert counter.total_count == 28
    assert counter.count == 9
    assert counter.threshold == 10


def test_process_tag_content() -> None:
    """Test the _process_tag_content method of SegmentedReviewLabeler."""
    from ypl.backend.llm.review import SegmentedReviewLabeler

    # Create a minimal instance just for testing the method
    labeler = SegmentedReviewLabeler(model="gpt-4o", llm=MagicMock())
    assert isinstance(labeler, SegmentedReviewLabeler)
    # Test case 1: Basic tag content extraction
    lines = [
        "<segment 1>",
        "This is some content",
        "across multiple lines",
        "</segment 1>",
    ]
    content, end_idx = labeler._process_tag_content(lines, 0, "segment")
    print(content)
    assert content == "This is some content\nacross multiple lines"
    assert end_idx == 3

    # Test case 2: Handling [insert verbatim] markers
    lines = [
        "<review 2>",
        "[insert verbatim review 2]",
        "The explanation here",
        "continues here",
        "</review 2>",
    ]
    content, end_idx = labeler._process_tag_content(lines, 0, "review")
    assert content == "The explanation here\ncontinues here"
    assert end_idx == 4

    # Test case 3: Preserving whitespace and empty lines
    lines = [
        "<updated-segment 3>",
        "  Indented line",
        "",
        "    Double indented",
        "</updated-segment 3>",
    ]
    content, end_idx = labeler._process_tag_content(lines, 0, "updated-segment")
    assert content == "  Indented line\n\n    Double indented"
    assert end_idx == 4

    lines = [
        "...",
        "...",
        "<updated-segment 3>",
        "  Indented line",
        "",
        "    Double indented",
        "</updated-segment 3>",
    ]
    content, end_idx = labeler._process_tag_content(lines, 2, "updated-segment")
    assert content == "  Indented line\n\n    Double indented"
    assert end_idx == 6


@pytest.mark.asyncio
@pytest.mark.repeat(3)
async def test_multi_delegate() -> None:
    class Op:
        # A base class for operations.
        def __init__(self, delay: float = 0.0) -> None:
            self.delay = delay

        def run(self, n: int) -> int:
            raise NotImplementedError

        async def run_async(self, n: int) -> int:
            return self.run(n)

    class MultiplyBy2(Op):
        def run(self, n: int) -> int:
            if self.delay > 0:
                time.sleep(self.delay)
            return n * 2

    class Add3(Op):
        def run(self, n: int) -> int:
            time.sleep(self.delay)
            return n + 3

    class RaiseIfInputIs3(Op):
        def run(self, n: int) -> int:
            if n == 3:
                raise ValueError("input is 3")
            return n

    delegator = Delegator(
        delegates={
            "mul_by_2": MultiplyBy2(delay=0.1),
            "add_3": Add3(delay=0.1),
            "raise_if_3": RaiseIfInputIs3(),
        }
    )

    # Normal run of sync methods
    assert await delegator.run(1) == {"mul_by_2": 2, "add_3": 4, "raise_if_3": 1}

    # Normal run of async methods
    assert await delegator.run_async(2) == {"mul_by_2": 4, "add_3": 5, "raise_if_3": 2}

    # One of the delegates raises an exception
    results = await delegator.run(3)
    assert results["mul_by_2"] == 6
    assert results["add_3"] == 6
    assert isinstance(results["raise_if_3"], ValueError)
    assert str(results["raise_if_3"]) == "input is 3"

    # First completed mode - everything except the fast-returning delegate should be cancelled
    delegator.priority_groups = [["mul_by_2"]]
    delegator.priority_groups_set = {"mul_by_2"}
    delegator.delegates["add_3"].delay = 0.2
    results = await delegator.run(3)
    assert results["mul_by_2"] == 6
    assert isinstance(results["add_3"], EarlyTerminatedException)
    assert isinstance(results["raise_if_3"], ValueError)

    # Check exception is raised if delegated method does not exist on one of the delegates
    delegator = Delegator(delegates={"mul_by_2": MultiplyBy2()})
    results = await delegator.non_existent_method()
    assert isinstance(results["mul_by_2"], AttributeError)

    # Test timeouts
    def setup_delegator(**kwargs: Any) -> Delegator:
        return Delegator(
            delegates={
                "add_3_fast_1": Add3(delay=0.0),
                "add_3_fast_2": Add3(delay=0.0),
                "add_3_medium": Add3(delay=0.3),
                "add_3_slow_1": Add3(delay=0.7),
                "add_3_slow_2": Add3(delay=0.7),
            },
            timeout_secs=0.5,
            **kwargs,
        )

    delegator = setup_delegator()
    results = await delegator.run(1)
    assert results["add_3_fast_1"] == results["add_3_fast_2"] == results["add_3_medium"] == 4
    assert isinstance(results["add_3_slow_1"], TimeoutError)
    assert isinstance(results["add_3_slow_2"], TimeoutError)

    # Timeouts on first completed mode: at least one fast delegate should returned, others delegates should be cancelled
    delegator = setup_delegator(early_terminate_on=["add_3_fast_1", "add_3_fast_2"])
    results = await delegator.run(1)
    assert results["add_3_fast_1"] == 4 or results["add_3_fast_2"] == 4
    assert isinstance(results["add_3_medium"], EarlyTerminatedException)
    assert isinstance(results["add_3_slow_1"], EarlyTerminatedException)
    assert isinstance(results["add_3_slow_2"], EarlyTerminatedException)

    delegator = setup_delegator(early_terminate_on=["add_3_medium"])
    results = await delegator.run(1)
    assert results["add_3_fast_1"] == 4
    assert results["add_3_fast_2"] == 4
    assert results["add_3_medium"] == 4
    assert isinstance(results["add_3_slow_1"], EarlyTerminatedException)
    assert isinstance(results["add_3_slow_2"], EarlyTerminatedException)

    # with multiple priority groups, the fast one will finish first but they need to wait for
    # medium and slow ones.
    delegator = setup_delegator(
        priority_groups=[["add_3_slow_1", "add_3_medium"], ["add_3_fast_1"], ["add_3_fast_2", "add_3_slow_2"]]
    )
    results = await delegator.run(1)
    assert isinstance(results["add_3_slow_1"], EarlyTerminatedException)
    assert results["add_3_medium"] == 4
    assert results["add_3_fast_1"] == 4 or results["add_3_fast_2"] == 4
    assert isinstance(results["add_3_slow_2"], EarlyTerminatedException)

    # another test where not all delegates are in the priority groups
    delegator = setup_delegator(priority_groups=[["add_3_slow_1", "add_3_slow_2"], ["add_3_medium"]])
    results = await delegator.run(1)
    assert isinstance(results["add_3_slow_1"], TimeoutError) or isinstance(results["add_3_slow_2"], TimeoutError)
    assert results["add_3_medium"] == 4
    assert results["add_3_fast_1"] == 4
    assert results["add_3_fast_2"] == 4


def test_extract_json_dict_from_text() -> None:
    assert extract_json_dict_from_text('```json{id = 1, name="xyz"}```') == '{id = 1, name="xyz"}'
    assert extract_json_dict_from_text('{id = 1, name="xyz"}random_text') == '{id = 1, name="xyz"}'
    assert extract_json_dict_from_text("no braces in text") == "no braces in text"
    assert extract_json_dict_from_text("no starting brace }") == "no starting brace }"
    assert extract_json_dict_from_text("{ no ending brace") == "{ no ending brace"
    assert extract_json_dict_from_text("in reverse order } {") == "in reverse order } {"
    assert (
        extract_json_dict_from_text(
            """
                Nested multi-line json:
                {
                    "id": 1,
                    "name": "xyz",
                    "nested": {
                        "id": 2,
                        "name": "abc"
                    }
                }
        """
        )
        == """{
                    "id": 1,
                    "name": "xyz",
                    "nested": {
                        "id": 2,
                        "name": "abc"
                    }
                }"""
    )


class MockSession:
    def __init__(self) -> None:
        self.exec = MagicMock()
        self.add = MagicMock()
        self.commit = MagicMock()
        self.refresh = MagicMock()
        self.delete = MagicMock()
        self.get = MagicMock()

    def __enter__(self) -> "MockSession":
        return self

    def __exit__(self, _: Any, __: Any, ___: Any) -> None:
        pass
