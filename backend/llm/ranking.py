from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

ELO_INIT_RATING = 1000.0
ELO_K = 4
ELO_BASE = 10
ELO_SCALE = 400


@dataclass
class Battle:
    model_a: str
    model_b: str
    # Convention is between [0..1], where 0 means "loss" and 1 means "win".
    result: float


@dataclass
class AnnotatedFloat:
    """An annotated value."""
    value: float | None
    annotation: str

    def __float__(self) -> float | None:
        return self.value


class Ranker:
    def predict(self, model_a: str, model_b: str) -> AnnotatedFloat:
        """Predict the likely outcome of a battle between `model_a` and `model_b`."""
        raise NotImplementedError

    def update(self, model_a: str, model_b: str, result: float) -> None:
        """Update the ranker with a result of a battle."""
        raise NotImplementedError

    def rank(self, model: str) -> AnnotatedFloat:
        """Return the relative rank of a model, compared to others."""
        raise NotImplementedError


class DataFrameRanker(Ranker):
    """A Ranker that uses a pd.DataFrame for the underlying data."""

    def __init__(self, battles: Iterable[Battle] = ()):
        self.battles = pd.DataFrame(columns=["model_a", "model_b", "result"])
        for battle in battles:
            self.update(battle.model_a, battle.model_b, battle.result)

    def update(self, model_a: str, model_b: str, result: float) -> None:
        if not (0 <= result <= 1):
            raise ValueError(f"Result must be between 0 and 1, got {result}")

        battle = pd.DataFrame([[model_a, model_b, result]], columns=self.battles.columns)
        if self.battles.empty:
            # Not strictly required, but avoids a pandas FutureWarning.
            self.battles = battle
        else:
            self.battles = pd.concat([self.battles, battle], ignore_index=True)


class NaiveRanker(DataFrameRanker):
    """A ranker that uses simple history stats."""

    def _count_and_sum_results(self, model_a: str, model_b: str, reverse: bool = False) -> tuple[int, float]:
        """Counts the number of battles between the models and sums the results.

        @param model_a: The model to count and sum the results for.
        @param model_b: The model to count and sum the results for.
        @param reverse: Whether to reverse the result (take the result of model_b vs model_a).
        @return: The number of battles and the sum of the results.
        """
        df = (self.battles["model_a"] == model_a) & (self.battles["model_b"] == model_b)
        results = self.battles.loc[df]["result"]
        if reverse:
            results = 1 - results
        return (0, None) if df.empty else (df.sum(), results.sum())

    def rank(self, model: str) -> AnnotatedFloat:
        """Returns the sum of results in the battles `model` participated in, normalized by their count."""
        num_battles = ((self.battles.model_a == model) | (self.battles.model_b == model)).sum()
        if num_battles == 0:
            return AnnotatedFloat(None, "No battles")

        total_score = (
            self.battles[self.battles.model_a == model].result.sum()
            + (1 - self.battles[self.battles.model_b == model].result).sum()
        )

        return AnnotatedFloat(value=total_score / num_battles, annotation=f"{total_score=}, {num_battles=}")

    def predict(self, model_a: str, model_b: str) -> AnnotatedFloat:
        """Returns the fraction of battles `model_a` won over `model_b`."""

        # Ignore the order of the models within a battle.
        battles_a_b = self._count_and_sum_results(model_a, model_b)
        battles_b_a = self._count_and_sum_results(model_b, model_a, reverse=True)
        num_battles = battles_a_b[0] + battles_b_a[0]

        sum_results: float | None = None
        annotation = f"[{model_a} vs {model_b}]: {battles_a_b[0]} battles"
        if battles_a_b[1]:
            sum_results = battles_a_b[0]
        annotation += f"; [{model_b} vs {model_a}]: {battles_b_a[0]} battles"
        if battles_b_a[1]:
            if sum_results is None:
                sum_results = battles_b_a[1]
            else:
                sum_results += battles_b_a[1]
        if sum_results is not None:
            annotation += f"; sum({model_a})={sum_results:.2f}"
        result = sum_results / num_battles if sum_results else None

        return AnnotatedFloat(value=result, annotation=annotation)

    def __str__(self) -> str:
        return self.battles.to_string(index=False)


class EloRanker(DataFrameRanker):
    """A ranker that uses the Elo rating system."""

    def __init__(
        self,
        battles: Iterable[Battle] = (),
        init_rating: float = ELO_INIT_RATING,
        k: int = ELO_K,
        base: int = ELO_BASE,
        scale: int = ELO_SCALE,
    ):
        """Initialize the ranker.

        @param battles: The battles to use.
        @param init_rating: The initial rating for a model.
        @param k: The k factor for the Elo rating system; higher values mean more sensitive to recent wins.
        @param base: The base for the Elo rating system.
        @param scale: The scale for the Elo rating system.
        """
        super().__init__(battles)
        self.init_rating = init_rating
        self.k = k
        self.base = base
        self.scale = scale
        # Maps models to their rating.
        self.ratings: dict[str, float] = defaultdict(lambda: self.init_rating)
        # Maps models to the list of score adjustments from the initial rating.
        self.adjustments: dict[str, list[float]] = defaultdict(list)

    def update(self, model_a: str, model_b: str, result: float) -> None:
        # Log the battle.
        super().update(model_a, model_b, result)

        rating_a = self.ratings[model_a]
        rating_b = self.ratings[model_b]
        expected_a = 1 / (1 + self.base ** ((rating_b - rating_a) / self.scale))
        expected_b = 1 / (1 + self.base ** ((rating_a - rating_b) / self.scale))

        # The amount that needs to be changed per model.
        adjustment_a = self.k * (result - expected_a)
        adjustment_b = self.k * (1 - result - expected_b)
        self.ratings[model_a] += adjustment_a
        self.ratings[model_b] += adjustment_b

        # Log the adjustments, for observability.
        self.adjustments[model_a].append(adjustment_a)
        self.adjustments[model_b].append(adjustment_b)

    def rank(self, model: str) -> AnnotatedFloat:
        rank = self.ratings[model]

        # Construct an annotation.
        adjustments = self.adjustments[model]
        num_adjustments = len(adjustments)
        num_negative = np.sum(np.array(adjustments) < 0)
        mean_adjustment = np.mean(adjustments) if num_adjustments > 0 else 0
        stdev_adjustment = np.std(adjustments) if num_adjustments > 0 else 0
        annotation = (
            f"Starting value {self.init_rating}; {num_adjustments} adjustments ({num_negative} negative, "
            f"mean={mean_adjustment:.2f}, stdev={stdev_adjustment:.2f})"
        )
        return AnnotatedFloat(value=rank, annotation=annotation)

    def predict(self, model_a: str, model_b: str) -> AnnotatedFloat:
        """Predict the outcome of a battle between `model_a` and `model_b` as a sigmoid of the rating difference."""
        rating_a = self.ratings[model_a]
        rating_b = self.ratings[model_b]
        win_probability = 1 / (1 + self.base ** ((rating_b - rating_a) / self.scale))
        return AnnotatedFloat(value=win_probability, annotation=f"{rating_a=:.2f}, {rating_b=:.2f}")
