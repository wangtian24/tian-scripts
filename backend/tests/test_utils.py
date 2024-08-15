from collections.abc import Iterable

import numpy as np
from pytest import approx, raises

from backend.llm.utils import norm_softmax


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
