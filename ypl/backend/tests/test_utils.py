from collections.abc import Iterable
from typing import Any
from unittest.mock import MagicMock

import numpy as np
from pytest import approx, raises

from ypl.backend.llm.utils import ThresholdCounter, norm_softmax


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
