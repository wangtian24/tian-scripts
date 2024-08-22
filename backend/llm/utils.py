import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

EPSILON = 1e-9


def combine_short_sentences(
    sentences: list[str], max_combined_length: int = 40, max_single_length: int = 10
) -> list[str]:
    """Combine short sentences into a single sentence."""
    combined_sentences = []
    current_sentence = ""

    for sentence in sentences:
        if (len(current_sentence) + len(sentence) <= max_combined_length) or (
            len(current_sentence) < max_single_length
        ):
            current_sentence += " " + sentence if current_sentence else sentence
        else:
            if current_sentence:
                combined_sentences.append(current_sentence.strip())
            current_sentence = sentence

    if current_sentence:
        combined_sentences.append(current_sentence.strip())

    return combined_sentences


@dataclass
class Battle:
    model_a: str
    model_b: str
    # Convention is between [0..1], where 0 means "loss" and 1 means "win".
    result_a: float

    def winner(self) -> str | None:
        return self.model_a if self.result_a > 0.5 else self.model_b if self.result_a < 0.5 else None

    def loser(self) -> str | None:
        return self.model_b if self.result_a > 0.5 else self.model_a if self.result_a < 0.5 else None

    def tie(self) -> bool:
        return self.result_a == 0.5


@dataclass
class AnnotatedFloat:
    """An annotated value."""

    value: float | None
    annotation: str | None

    def __float__(self) -> float | None:
        return self.value


@dataclass
class RatedModel:
    """A model with a rating."""

    model: str
    rating: AnnotatedFloat

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "rating": self.rating.value, "annotation": self.rating.annotation}


def norm_softmax(arr: Iterable[float]) -> np.ndarray:
    """
    Returns pseudo-probabilities using a sigmoid-like normalization and softmax.
    """
    arr = np.array(arr, dtype=float)

    if any(np.isnan(x) or np.isinf(x) for x in arr):
        raise ValueError("Input array contains infinite or NaN values.")

    if len(arr) <= 1:
        return np.array([1] * len(arr))

    if np.all(arr == arr[0]):
        return np.full_like(arr, 1 / len(arr))  # Uniform distribution

    scale = max(np.abs(arr).max(), EPSILON)
    sigmoid = 1 / (1 + np.exp(-arr / scale))
    exp_sigmoid = np.exp(sigmoid)

    return np.array(exp_sigmoid / np.sum(exp_sigmoid))


@dataclass
class ThresholdCounter:
    """A counter that tracks updates and determines when a threshold is reached."""

    # Number of times the counter was incremented since last reset.
    count: int = 0
    # Total number of times the counter was incremented.
    total_count: int = 0
    # The threshold at which the count is considered to be reached.
    threshold: int = 1
    # The maximum threshold value.
    max_threshold: int = 25000
    # The rate at which the threshold grows every time it is reset.
    growth_rate: float = 1.2345
    # Number of times the threshold was reset.
    reset_count: int = 0

    def increment(self) -> None:
        self.count += 1
        self.total_count += 1

    def is_threshold_reached(self) -> bool:
        return self.count >= self.threshold

    def reset(self) -> None:
        """Resets the counter and increases the threshold."""
        self.count = 0
        self.threshold = min(math.ceil(self.threshold * self.growth_rate), self.max_threshold)
        self.reset_count += 1
