from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import cache
from typing import Any, ClassVar, Literal

import choix
import numpy as np
import pandas as pd

from backend.llm.utils import AnnotatedFloat, Battle, RankedModel

# Rating of a new model.
ELO_INIT_RATING = 1000.0
# The k factor for the Elo rating system; higher values mean more sensitive to recent wins.
ELO_K = 4
# The exponent base for the Elo rating system.
ELO_BASE = 10
# The scale for the Elo rating system.
ELO_SCALE = 400

# The default cost for a model.
DEFAULT_COST = 1.0

# Algorithms available for Choice Axiom ranking (Bradley Terry-like). See https://choix.lum.li/ for details.
CHOIX_RANKER_ALGORITHMS = (
    "ilsr_pairwise",
    "lsr_pairwise",
    "mm_pairwise",
    "opt_pairwise",
    "rank_centrality",
)

# The default algorithm for Choice Axiom ranking.
CHOIX_DEFAULT_ALGORITHM = "ilsr_pairwise"

ConfInterval = tuple[float, float]


class Ranker:
    """A relative ranker for LLM models."""

    # A name for the ranker.
    ranker_type: ClassVar[str]

    def __init__(self, models: list[Any] | None = None, costs: list[float] | None = None):
        """Initialize the ranker.

        Args:
            models: The models to use.
            costs: The costs of the models, should match the number and order of `models`.
        """
        if not models:
            models = []
        if not costs:
            self.costs = [DEFAULT_COST] * len(models)
        else:
            assert len(models) == len(costs), f"Number of models ({len(models)}) and costs ({len(costs)}) differ"
            self.costs = costs
        self.models = models

    def predict(self, model_a: str, model_b: str) -> float | None:
        """Predict the likely outcome of a battle between `model_a` and `model_b`."""
        raise NotImplementedError

    def predict_annotate(self, model_a: str, model_b: str) -> AnnotatedFloat:
        """Predict the likely outcome of a battle between `model_a` and `model_b`, with an explanation."""
        return AnnotatedFloat(
            value=self.predict(model_a, model_b), annotation=self.annotate_prediction(model_a, model_b)
        )

    def update(self, model_a: str, model_b: str, result: float) -> None:
        """Update the ranker with a result of a battle."""
        raise NotImplementedError

    def rank(self, model: str) -> float | None:
        """Return the relative rank of a model, compared to others."""
        raise NotImplementedError

    def rank_annotate(self, model: str) -> AnnotatedFloat:
        """Return the relative rank of a model, compared to others, with an explanation."""
        return AnnotatedFloat(value=self.rank(model), annotation=self.annotate_model(model))

    def ranks(self) -> dict[str, float]:
        """Return the ranks of all models."""
        raise NotImplementedError

    def annotate_model(self, model: str) -> str:
        """Return an annotation for a model."""
        raise NotImplementedError

    def annotate_prediction(self, model_a: str, model_b: str) -> str:
        """Return an annotation for a prediction."""
        raise NotImplementedError

    def leaderboard(self) -> list[RankedModel]:
        """Return the leaderboard."""
        return [RankedModel(model=model, rank=self.rank_annotate(model)) for model in self.models]

    def add_model(self, model: str, cost: float = DEFAULT_COST) -> None:
        """Add a model to the ranker."""
        self.models.append(model)
        self.costs.append(cost)


class DataFrameRanker(Ranker):
    """A Ranker that uses a pd.DataFrame for the underlying data."""

    def __init__(
        self, models: list[Any] | None = None, costs: list[float] | None = None, battles: Iterable[Battle] = ()
    ):
        """Initialize the ranker.

        Args:
            models: The models to use.
            costs: The costs of the models, should match the number and order of `models`.
            battles: The battles to use.
        """
        super().__init__(models, costs)
        self.battles = pd.DataFrame(columns=["model_a", "model_b", "result_a"])
        for battle in battles:
            self.update(battle.model_a, battle.model_b, battle.result_a)

    def update(self, model_a: str, model_b: str, result_a: float) -> None:
        """Update the ranker with a result of a battle.

        Args:
            model_a: The first model in the battle.
            model_b: The second model in the battle.
            result_a: The result of the battle from model_a's perspective -- 0 for loss, 1 for win.
        """
        if not (0 <= result_a <= 1):
            raise ValueError(f"Result must be between 0 and 1, got {result_a}")

        models = set(self.models)
        if model_a not in models:
            raise ValueError(f"Model {model_a} not found; please add it first")
        if model_b not in models:
            raise ValueError(f"Model {model_b} not found; please add it first")

        battle = pd.DataFrame([[model_a, model_b, result_a]], columns=self.battles.columns)
        if self.battles.empty:
            # Not strictly required, but avoids a pandas FutureWarning.
            self.battles = battle
        else:
            self.battles = pd.concat([self.battles, battle], ignore_index=True)


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

    def _get_num_battles(self, model: str) -> int:
        return ((self.battles.model_a == model) | (self.battles.model_b == model)).sum()

    def _get_total_score(self, model: str) -> float:
        return float(
            self.battles[self.battles.model_a == model].result_a.sum()
            + (1 - self.battles[self.battles.model_b == model].result_a).sum()
        )

    def annotate_model(self, model: str) -> str:
        """Return an annotation for a model."""
        num_battles = self._get_num_battles(model)
        total_score = self._get_total_score(model)

        return f"{total_score=}, {num_battles=}"

    def rank(self, model: str) -> float | None:
        """Returns the sum of results in the battles `model` participated in, normalized by their count."""
        num_battles = self._get_num_battles(model)
        if num_battles == 0:
            return None

        total_score = self._get_total_score(model)

        return total_score / num_battles

    def annotate_prediction(self, model_a: str, model_b: str) -> str:
        # Ignore the order of the models within a battle.
        battles_a_b = self._count_and_sum_results(model_a, model_b)
        battles_b_a = self._count_and_sum_results(model_b, model_a, reverse=True)

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

        return annotation

    def predict(self, model_a: str, model_b: str) -> float | None:
        """Returns the fraction of battles `model_a` won over `model_b`."""

        # Ignore the order of the models within a battle.
        battles_a_b = self._count_and_sum_results(model_a, model_b)
        battles_b_a = self._count_and_sum_results(model_b, model_a, reverse=True)
        num_battles = battles_a_b[0] + battles_b_a[0]

        sum_results: float | None = None
        if battles_a_b[1]:
            sum_results = battles_a_b[0]
        if battles_b_a[1]:
            if sum_results is None:
                sum_results = battles_b_a[1]
            else:
                sum_results += battles_b_a[1]
        result = sum_results / num_battles if sum_results else None

        return result

    def __str__(self) -> str:
        return self.battles.to_string(index=False)


class EloRanker(DataFrameRanker):
    """A ranker that uses the Elo rating system."""

    ranker_type = "elo"

    def __init__(
        self,
        models: list[Any] | None = None,
        costs: list[float] | None = None,
        battles: Iterable[Battle] = (),
        init_rating: float = ELO_INIT_RATING,
        k: int = ELO_K,
        base: int = ELO_BASE,
        scale: int = ELO_SCALE,
    ):
        """Initialize the ranker.

        Args:
            models: The models to use.
            costs: The costs of the models, should match the number and order of `models`.
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
        super().__init__(models, costs, battles)

    def update(self, model_a: str, model_b: str, result: float) -> None:
        """Update the ranker with a result of a battle."""
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

    def rank(self, model: str) -> float | None:
        if model not in self.ratings:
            self.add_model(model, DEFAULT_COST)
            self.ratings[model] = self.init_rating
        return self.ratings.get(model)

    def ranks(self) -> dict[str, float]:
        return self.ratings

    def annotate_model(self, model: str) -> str:
        adjustments = self.adjustments[model]
        num_adjustments = len(adjustments)
        num_negative = np.sum(np.array(adjustments) < 0)
        mean_adjustment = np.mean(adjustments) if num_adjustments > 0 else 0
        stdev_adjustment = np.std(adjustments) if num_adjustments > 0 else 0
        return (
            f"Starting value {self.init_rating}; {num_adjustments} adjustments ({num_negative} negative, "
            f"mean={mean_adjustment:.2f}, stdev={stdev_adjustment:.2f})"
        )

    def annotate_prediction(self, model_a: str, model_b: str) -> str:
        rating_a = self.ratings[model_a]
        rating_b = self.ratings[model_b]
        return f"{rating_a=:.2f}, {rating_b=:.2f}"

    def predict(self, model_a: str, model_b: str) -> float | None:
        """Predict the outcome of a battle between `model_a` and `model_b` as a sigmoid of the rating difference."""
        rating_a = self.rank(model_a)
        rating_b = self.rank(model_b)
        if rating_a is None or rating_b is None:
            return None
        win_probability = 1.0 / (1 + self.base ** ((rating_b - rating_a) / self.scale))
        return float(win_probability)


class ChoixRanker(Ranker):
    """A ranker that uses various algorithms for Luce's Choice Axiom from the Choix library.

    See https://choix.lum.li/.
    """

    ranker_type = "choix"

    def __init__(
        self,
        models: list[Any] | None = None,
        costs: list[float] | None = None,
        battles: Iterable[Battle] = (),
        tie_policy: Literal["ignore", "add_twice"] = "add_twice",
        choix_ranker_algorithm: str = CHOIX_DEFAULT_ALGORITHM,
        choix_alpha: float = 0.001,
        choix_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the ranker.
        Args:
            models: The models to use.
            costs: The costs of the models.
            battles: The battles to use.
            tie_policy: The policy to use when a tie occurs: either ignore or add the battle twice.
            choix_ranker_algorithm: The algorithm to use for ranking.
            choix_alpha: The regularization parameter for the Choix ranker.
            choix_kwargs: Keyword arguments to pass to the ranker.
        """
        if choix_ranker_algorithm not in CHOIX_RANKER_ALGORITHMS:
            raise ValueError(f"Invalid ranker: '{choix_ranker_algorithm}'; valid rankers: {CHOIX_RANKER_ALGORITHMS}")
        super().__init__(models, costs)
        # Choix represents models as running ints starting from 0; we map models to these ints here.
        self.model_ids: dict[str, int] = {}
        if models:
            self.model_ids = {model: i for i, model in enumerate(models)}
        # A list of battles, where each battle is a pair of model IDs.
        self.battles: list[tuple[int, int]] = []
        self.tie_policy = tie_policy
        # Maps a model to its most recently calculated rank.
        self.ratings: dict[str, float] = {}
        # Maps a model to the number of wins, losses, and ties it has; used for rank annotations.
        self.wins: dict[str, int] = Counter()
        self.losses: dict[str, int] = Counter()
        self.ties: dict[str, int] = Counter()
        # True if the data in `self.ranks` needs to be updated.
        self.ratings_are_stale = True
        # The parameters for the underlying Choix ranker.
        self.choix_params: list[float] = []
        self.choix_ranker = getattr(choix, choix_ranker_algorithm)
        self.choix_alpha = choix_alpha
        self.choix_kwargs = choix_kwargs or {}
        for battle in battles:
            self.update(battle.model_a, battle.model_b, battle.result_a)

    def add_model(self, model: str, cost: float = DEFAULT_COST) -> None:
        """Add a model to the ranker."""
        super().add_model(model, cost)
        self.model_ids[model] = len(self.model_ids)

    def update(self, model_a: str, model_b: str, result_a: float) -> None:
        """Update the ranker with a result of a battle."""
        self.ratings_are_stale = True
        # Assign new model IDs if needed.
        for model in [model_a, model_b]:
            if model not in self.model_ids:
                self.add_model(model, DEFAULT_COST)

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

    def rank(self, model: str) -> float | None:
        """Returns the rank of `model`."""
        self._maybe_update_ranks()
        return self.ratings.get(model)

    def ranks(self) -> dict[str, float]:
        """Returns the ranks of all models."""
        self._maybe_update_ranks()
        return self.ratings

    def annotate_model(self, model: str) -> str:
        """Returns an annotation for `model`."""
        return f"Wins: {self.wins[model]}, Losses: {self.losses[model]}, Ties: {self.ties[model]}"

    def _maybe_update_ranks(self) -> None:
        """Recalculate ranks, if they are stale."""
        if not self.ratings_are_stale:
            return
        ids_to_models = {id: model for model, id in self.model_ids.items()}
        try:
            self.choix_params = self.choix_ranker(
                len(self.model_ids), self.battles, alpha=self.choix_alpha, **self.choix_kwargs
            )
            self.ratings = {ids_to_models[id]: rank for id, rank in enumerate(self.choix_params)}
        except IndexError:
            # Choix can fail when there are too few battles.
            self.ratings = {model: 0 for model in self.model_ids.keys()}
        self.ratings_are_stale = False

    def leaderboard(self) -> list[RankedModel]:
        self._maybe_update_ranks()
        return super().leaderboard()

    def predict(self, model_a: str, model_b: str) -> float | None:
        self._maybe_update_ranks()
        model_a_id = self.model_ids.get(model_a)
        model_b_id = self.model_ids.get(model_b)
        if model_a_id is None or model_b_id is None:
            raise ValueError(f"Model '{model_a}' or '{model_b}' not found")
        prob_a_wins, _ = choix.probabilities([model_a_id, model_b_id], self.choix_params)
        return float(prob_a_wins)

    def annotate_prediction(self, model_a: str, model_b: str) -> str:
        ranks = self.ranks()
        rank_a = ranks[model_a]
        rank_b = ranks[model_b]
        return f"{rank_a=:.2f}, {rank_b=:.2f}"


class ChoixRankerConfIntervals(ChoixRanker):
    """A ChoixRanker that adds confidence intervals to the predictions."""

    ranker_type = "choix_conf_intervals"

    def __init__(
        self,
        models: list[Any] | None = None,
        costs: list[float] | None = None,
        battles: Iterable[Battle] = (),
        tie_policy: Literal["ignore", "add_twice"] = "add_twice",
        choix_ranker_algorithm: str = "ilsr_pairwise",
        choix_alpha: float = 0.001,
        choix_kwargs: dict[str, Any] | None = None,
        num_bootstrap_iterations: int = 20,
        bootstrap_sample_fraction: float = 1.0,
        seed: int = 123,
    ):
        super().__init__(models, costs, battles, tie_policy, choix_ranker_algorithm, choix_alpha, choix_kwargs)
        self.conf_intervals: dict[str, ConfInterval] = {}
        self.num_bootstrap_iterations = num_bootstrap_iterations
        self.bootstrap_sample_fraction = bootstrap_sample_fraction
        self.seed = seed

    def rank_conf_intervals(self, model: str) -> tuple[float | None, float | None, float | None]:
        """Returns the rank of `model`, along with its confidence interval."""
        self._maybe_update_ranks()
        rank = self.rank(model)
        conf_intervals = self.conf_intervals.get(model, (None, None))
        return rank, conf_intervals[0], conf_intervals[1]

    def ranks_conf_intervals(self) -> dict[str, tuple[float, ConfInterval]]:
        """Returns the ranks of all models, along with their confidence intervals."""
        return {model: (rank, self.conf_intervals[model]) for model, rank in self.ranks().items()}

    def annotate_model(self, model: str) -> str:
        """Returns an annotation for `model`."""
        return (
            f"{super().annotate_model(model)} "
            f"({self.conf_intervals[model][0]:.3f} to {self.conf_intervals[model][1]:.3f})"
        )

    def _maybe_update_ranks(self) -> None:
        """Recalculate ranks, if they are stale."""
        if not self.ratings_are_stale:
            return

        # Use bootstrap to estimate the confidence intervals.
        ids_to_models = {id: model for model, id in self.model_ids.items()}
        bootstrap_ranks = []  # The ranks of the models after each bootstrap iteration.
        battles = np.array(self.battles)  # Convert battles to a numpy array for efficient sampling.
        num_bootstrap_samples = int(len(self.battles) * self.bootstrap_sample_fraction)
        rng = np.random.default_rng(self.seed)
        try:
            for _ in range(self.num_bootstrap_iterations):
                sampled_battles = rng.choice(battles, size=num_bootstrap_samples, replace=True)
                choix_params = self.choix_ranker(
                    len(self.model_ids), sampled_battles, alpha=self.choix_alpha, **self.choix_kwargs
                )
                bootstrap_ranks.append(choix_params)

            mean_ranks = np.mean(bootstrap_ranks, axis=0)
            lower_bounds = np.percentile(bootstrap_ranks, 5, axis=0)
            upper_bounds = np.percentile(bootstrap_ranks, 95, axis=0)

            # The ranker uses the average as the final rank.
            self.choix_params = mean_ranks
            self.ratings = {ids_to_models[id]: rank for id, rank in enumerate(mean_ranks)}
            self.conf_intervals = {
                ids_to_models[id]: (lower, upper)
                for id, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds, strict=True))
            }
        except IndexError:
            # Choix can fail when there are too few battles.
            self.ratings = {model: 0 for model in self.model_ids.keys()}
        self.ratings_are_stale = False


RANKER_CLASSES: tuple[type[Ranker], ...] = (EloRanker, NaiveRanker, ChoixRanker, ChoixRankerConfIntervals)
RANKER_TYPES: list[str] = [cls.ranker_type for cls in RANKER_CLASSES]


@cache
def get_ranker(ranker_type: str, *args: Any, **kwargs: Any) -> Ranker:
    """Returns a singleton instance of the ranker of type `ranker_type`."""
    for ranker_cls in RANKER_CLASSES:
        if ranker_type == ranker_cls.ranker_type:
            return ranker_cls(*args, **kwargs)
    raise ValueError(f"Unsupported ranking type: {ranker_type}")
