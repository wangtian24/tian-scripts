from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import cache
from typing import Any, ClassVar, Literal

import choix
import numpy as np
import pandas as pd

from backend.llm.utils import AnnotatedFloat, Battle, RankedModel

ELO_INIT_RATING = 1000.0
ELO_K = 4
ELO_BASE = 10
ELO_SCALE = 400

# See https://choix.lum.li/ for more details.
CHOIX_RANKER_ALGORITHMS = (
    "ilsr_pairwise",
    "lsr_pairwise",
    "mm_pairwise",
    "opt_pairwise",
    "rank_centrality",
)


class Ranker:
    """A relative ranker for LLM models."""

    # A name for the ranker.
    ranker_type: ClassVar[str]

    def predict(self, model_a: str, model_b: str) -> AnnotatedFloat:
        """Predict the likely outcome of a battle between `model_a` and `model_b`."""
        raise NotImplementedError

    def update(self, model_a: str, model_b: str, result: float) -> None:
        """Update the ranker with a result of a battle."""
        raise NotImplementedError

    def rank(self, model: str) -> AnnotatedFloat:
        """Return the relative rank of a model, compared to others."""
        raise NotImplementedError

    def leaderboard(self) -> list[RankedModel]:
        """Return the leaderboard."""
        raise NotImplementedError


class DataFrameRanker(Ranker):
    """A Ranker that uses a pd.DataFrame for the underlying data."""

    def __init__(self, battles: Iterable[Battle] = ()):
        self.battles = pd.DataFrame(columns=["model_a", "model_b", "result_a"])
        for battle in battles:
            self.update(battle.model_a, battle.model_b, battle.result_a)

    def update(self, model_a: str, model_b: str, result_a: float) -> None:
        if not (0 <= result_a <= 1):
            raise ValueError(f"Result must be between 0 and 1, got {result_a}")

        battle = pd.DataFrame([[model_a, model_b, result_a]], columns=self.battles.columns)
        if self.battles.empty:
            # Not strictly required, but avoids a pandas FutureWarning.
            self.battles = battle
        else:
            self.battles = pd.concat([self.battles, battle], ignore_index=True)

    def leaderboard(self) -> list[RankedModel]:
        """Return the leaderboard."""
        models = set(self.battles["model_a"].unique()) | set(self.battles["model_b"].unique())
        return [RankedModel(model, self.rank(model)) for model in models]


class NaiveRanker(DataFrameRanker):
    """A ranker that uses simple history stats."""

    ranker_type = "naive"

    def _count_and_sum_results(self, model_a: str, model_b: str, reverse: bool = False) -> tuple[int, float]:
        """Counts the number of battles between the models and sums the results.

        Args:
            model_a: The model to count and sum the results for.
            model_b: The model to count and sum the results for.
            reverse: Whether to reverse the result (take the result of model_b vs model_a).
        Returns:
            The number of battles and the sum of the results.
        """
        df = (self.battles["model_a"] == model_a) & (self.battles["model_b"] == model_b)
        results = self.battles.loc[df]["result_a"]
        if reverse:
            results = 1 - results
        return (0, None) if df.empty else (df.sum(), results.sum())

    def rank(self, model: str) -> AnnotatedFloat:
        """Returns the sum of results in the battles `model` participated in, normalized by their count."""
        num_battles = ((self.battles.model_a == model) | (self.battles.model_b == model)).sum()
        if num_battles == 0:
            return AnnotatedFloat(None, "No battles")

        total_score = (
            self.battles[self.battles.model_a == model].result_a.sum()
            + (1 - self.battles[self.battles.model_b == model].result_a).sum()
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

    ranker_type = "elo"

    def __init__(
        self,
        battles: Iterable[Battle] = (),
        init_rating: float = ELO_INIT_RATING,
        k: int = ELO_K,
        base: int = ELO_BASE,
        scale: int = ELO_SCALE,
    ):
        """Initialize the ranker.

        Args:
            battles: The battles to use.
            init_rating: The initial rating for a model.
            k: The k factor for the Elo rating system; higher values mean more sensitive to recent wins.
            base: The base for the Elo rating system.
            scale: The scale for the Elo rating system.
        """
        self.init_rating = init_rating
        self.k = k
        self.base = base
        self.scale = scale
        # Maps models to their rating.
        self.ratings: dict[str, float] = defaultdict(lambda: self.init_rating)
        # Maps models to the list of score adjustments from the initial rating.
        self.adjustments: dict[str, list[float]] = defaultdict(list)
        super().__init__(battles)

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

        # Collect data for an annotation.
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


class ChoixRanker(Ranker):
    """A ranker that uses various algorithms for Luce's Choice Axiom from the Choix library.

    See https://choix.lum.li/.
    """

    ranker_type = "choix"

    def __init__(
        self,
        battles: Iterable[Battle] = (),
        tie_policy: Literal["ignore", "add_twice"] = "add_twice",
        choix_ranker_algorithm: str = "ilsr_pairwise",
        choix_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the ranker.
        Args:
            battles: The battles to use.
            tie_policy: The policy to use when a tie occurs: either ignore or add the battle twice.
            choix_ranker_algorithm: The algorithm to use for ranking.
            choix_kwargs: Keyword arguments to pass to the ranker.
        """
        if choix_ranker_algorithm not in CHOIX_RANKER_ALGORITHMS:
            raise ValueError(f"Invalid ranker: '{choix_ranker_algorithm}'; valid rankers: {CHOIX_RANKER_ALGORITHMS}")
        # Choix represents models as running ints starting from 0; we map models to these ints here.
        self.model_ids: dict[str, int] = {}
        # A list of battles, where each battle is a pair of model IDs.
        self.battles: list[tuple[int, int]] = []
        self.tie_policy = tie_policy
        # Maps a model to its most recently calculated rank.
        self.ranks: dict[str, float] = {}
        # Maps a model to the number of wins, losses, and ties it has; used for rank annotations.
        self.wins: dict[str, int] = Counter()
        self.losses: dict[str, int] = Counter()
        self.ties: dict[str, int] = Counter()
        # True if the data in `self.ranks` needs to be updated.
        self.ranks_are_stale = True
        # The parameters for the underlying Choix ranker.
        self.choix_params: list[float] = []
        self.choix_ranker = getattr(choix, choix_ranker_algorithm)
        self.choix_kwargs = choix_kwargs or {}
        for battle in battles:
            self.update(battle.model_a, battle.model_b, battle.result_a)

    def update(self, model_a: str, model_b: str, result_a: float) -> None:
        self.ranks_are_stale = True
        # Assign new model IDs if needed.
        if model_a not in self.model_ids:
            self.model_ids[model_a] = len(self.model_ids)
        if model_b not in self.model_ids:
            self.model_ids[model_b] = len(self.model_ids)

        model_a_id = self.model_ids[model_a]
        model_b_id = self.model_ids[model_b]
        if result_a > 0.5:
            self.battles.append((model_a_id, model_b_id))
            self.wins[model_a] += 1
            self.losses[model_b] += 1
        elif result_a < 0.5:
            self.battles.append((model_b_id, model_a_id))
            self.wins[model_b] += 1
            self.losses[model_a] += 1
        else:
            self.ties[model_a] += 1
            self.ties[model_b] += 1
            if self.tie_policy == "ignore":
                return
            elif self.tie_policy == "add_twice":
                self.battles.append((model_a_id, model_b_id))
                self.battles.append((model_b_id, model_a_id))

    def rank(self, model: str) -> AnnotatedFloat:
        self._maybe_update_ranks()
        if model not in self.ranks:
            rank, annotation = 0.0, "No battles"
        else:
            rank, annotation = self.ranks[model], self.annotate(model)
        return AnnotatedFloat(value=rank, annotation=annotation)

    def annotate(self, model: str) -> str:
        return f"Wins: {self.wins[model]}, Losses: {self.losses[model]}, Ties: {self.ties[model]}"

    def _maybe_update_ranks(self) -> None:
        """Recalculate ranks, if they are stale."""
        if not self.ranks_are_stale:
            return
        ids_to_models = {id: model for model, id in self.model_ids.items()}
        try:
            self.choix_params = self.choix_ranker(len(self.model_ids), self.battles, **self.choix_kwargs)
            self.ranks = {ids_to_models[id]: rank for id, rank in enumerate(self.choix_params)}
        except IndexError:
            # Choix can fail when there are too few battles.
            self.ranks = {model: 0 for model in self.model_ids.keys()}
        self.ranks_are_stale = False

    def leaderboard(self) -> list[RankedModel]:
        self._maybe_update_ranks()
        return sorted(
            [
                RankedModel(model, AnnotatedFloat(value=rank, annotation=self.annotate(model)))
                for model, rank in self.ranks.items()
            ],
            key=lambda x: x.rank.value if x.rank.value is not None else float("-inf"),
            reverse=True,
        )

    def predict(self, model_a: str, model_b: str) -> AnnotatedFloat:
        self._maybe_update_ranks()
        model_a_id = self.model_ids.get(model_a)
        model_b_id = self.model_ids.get(model_b)
        if model_a_id is None or model_b_id is None:
            raise ValueError(f"Model '{model_a}' or '{model_b}' not found")
        prob_a_wins, _ = choix.probabilities([model_a_id, model_b_id], self.choix_params)
        rank_a = self.ranks[model_a]
        rank_b = self.ranks[model_b]
        annotation = f"{rank_a=:.2f}, {rank_b=:.2f}"
        return AnnotatedFloat(value=prob_a_wins, annotation=annotation)


RANKER_CLASSES: tuple[type[Ranker], ...] = (EloRanker, NaiveRanker, ChoixRanker)
RANKER_TYPES: list[str] = [cls.ranker_type for cls in RANKER_CLASSES]


@cache
def get_ranker(ranker_type: str, *args: Any, **kwargs: Any) -> Ranker:
    """Returns a singleton instance of the ranker of type `ranker_type`."""
    for ranker_cls in RANKER_CLASSES:
        if ranker_type == ranker_cls.ranker_type:
            return ranker_cls(*args, **kwargs)
    raise ValueError(f"Unsupported ranking type: {ranker_type}")
