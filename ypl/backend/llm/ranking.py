import concurrent
import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime
from functools import cache
from typing import Any, ClassVar, Literal

import choix
import numpy as np
import pandas as pd
from sqlalchemy import alias, func
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlmodel import Session, select
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ypl.backend.db import get_engine
from ypl.backend.llm.utils import (
    AnnotatedFloat,
    Battle,
    RatedModel,
    ThresholdCounter,
    fetch_categories_with_descriptions_from_db,
)
from ypl.db.chats import Chat, ChatMessage, Eval, EvalType, LanguageCode, MessageType, Turn, User
from ypl.db.language_models import LanguageModel
from ypl.db.ratings import OVERALL_CATEGORY_NAME, Category, Rating, RatingHistory
from ypl.logger import logger

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
CHOIX_DEFAULT_ALGORITHM = "lsr_pairwise"

DEFAULT_RANKER_TYPE = "choix_conf_intervals"
DEFAULT_RANKER_KWARGS = dict(choix_ranker_algorithm=CHOIX_DEFAULT_ALGORITHM)

MAX_NUM_WORKERS = 8
MIN_NUM_WORKERS = 1
BATTLES_PER_WORKER = 100000

ConfInterval = tuple[float, float]


def _score_to_elo(
    score: float,
    scale: float = ELO_SCALE,
    init_rating: float = ELO_INIT_RATING,
) -> float:
    """Convert a score to the Elo range."""
    return score * scale + init_rating


def _elo_to_probabilities(
    ratings: dict[str, float],
    scale: float = ELO_SCALE,
    base: float = ELO_BASE,
) -> dict[str, float]:
    """Convert Elo ratings to probabilities."""
    likelihoods = {model: base ** (score / scale) for model, score in ratings.items()}
    total_likelihood = sum(likelihoods.values())
    return {model: likelihood / total_likelihood for model, likelihood in likelihoods.items()}


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
        if models is None:
            models = []
        if not costs:
            self.costs = {m: DEFAULT_COST for m in models}
        else:
            assert len(models) == len(costs), f"Number of models ({len(models)}) and costs ({len(costs)}) differ"
            self.costs = dict(zip(models, costs, strict=True))
        self.models = set(models)
        # Maps a model to the number of its wins, losses, and ties.
        self.wins: dict[str, int] = Counter()
        self.losses: dict[str, int] = Counter()
        self.ties: dict[str, int] = Counter()
        self.total_battles = 0

    def get_total_battles(self) -> int:
        """Return the total number of battles."""
        return self.total_battles

    def get_models(self) -> set[str]:
        """Return the models."""
        return self.models

    def predict(self, model_a: str, model_b: str) -> float | None:
        """Predict the likely outcome of a battle between `model_a` and `model_b`."""
        raise NotImplementedError

    def predict_annotate(self, model_a: str, model_b: str) -> AnnotatedFloat:
        """Predict the likely outcome of a battle between `model_a` and `model_b`, with an explanation."""
        return AnnotatedFloat(
            value=self.predict(model_a, model_b), annotation=self.annotate_prediction(model_a, model_b)
        )

    def update(self, model_a: str, model_b: str, result_a: float, category: str | None = None) -> None:
        """Update the ranker with a result of a battle."""
        if result_a > 0.5:
            self.wins[model_a] += 1
            self.losses[model_b] += 1
        elif result_a < 0.5:
            self.wins[model_b] += 1
            self.losses[model_a] += 1
        else:
            self.ties[model_a] += 1
            self.ties[model_b] += 1
        self.total_battles += 1

    def get_rating(self, model: str) -> float | None:
        """Return the rating of a model."""
        raise NotImplementedError

    def get_annotated_rating(self, model: str) -> AnnotatedFloat:
        """Return the rating of a model with an explanation."""
        return AnnotatedFloat(value=self.get_rating(model), annotation=self.annotate_model(model))

    def get_ratings(self) -> dict[str, float]:
        """Return the ratings of all models."""
        raise NotImplementedError

    def get_probabilities(self) -> dict[str, float]:
        """Return the probabilities of all models."""
        raise NotImplementedError

    def annotate_model(self, model: str) -> str:
        """Return an annotation for a model."""
        raise NotImplementedError

    def annotate_prediction(self, model_a: str, model_b: str) -> str:
        """Return an annotation for a prediction."""
        raise NotImplementedError

    def get_rated_models(self) -> list[RatedModel]:
        """Return the rated models."""
        return [self.get_rated_model(model) for model in self.models]

    def get_rated_model(self, model: str) -> RatedModel:
        rating = self.get_rating(model)
        if rating is None:
            rating = 0
        return RatedModel(
            model=model,
            rating=rating,
            wins=self.wins[model],
            losses=self.losses[model],
            ties=self.ties[model],
            annotation=self.annotate_model(model),
        )

    def leaderboard(self) -> list[RatedModel]:
        """Return the leaderboard."""
        return sorted(
            self.get_rated_models(),
            key=lambda x: x.rating if x.rating is not None else float("-inf"),
            reverse=True,
        )

    def add_model(self, model: str, cost: float = DEFAULT_COST) -> None:
        """Add a model to the ranker."""
        self.models.add(model)
        self.costs[model] = cost

    def to_db(self, category_name: str | None = None, snapshot_timestamp: datetime | None = None) -> None:
        """Add the ratings to the database."""
        raise NotImplementedError

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.1),
        after=after_log(logger, logging.WARNING),
        retry=retry_if_exception_type((OperationalError, DatabaseError)),
    )
    def add_evals_from_db(
        self,
        category_names: list[str] | None = None,
        exclude_ties: bool = False,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        user_from_date: datetime | None = None,
        user_to_date: datetime | None = None,
        language_codes: list[str] | None = None,
    ) -> int:
        """Initialize the ranker with evals from the database, returning the number of evals added.

        Args:
            category_names: The categories to filter by.
            exclude_ties: Whether to exclude ties.
            from_date: Prompt created at or after this date.
            to_date: Prompt created at or before this date.
            user_from_date: User created at or after this date.
            user_to_date: User created at or before this date.
            language_codes: The language codes to filter by.

        Returns:
            The number of evals added.
        """
        ChatMessage1 = alias(ChatMessage)  # type: ignore
        ChatMessage2 = alias(ChatMessage)  # type: ignore

        query = (
            select(
                Eval.score_1,  # type: ignore
                Eval.eval_type,
                User.email,
                ChatMessage1.c.assistant_model_name.label("model_a"),
                ChatMessage2.c.assistant_model_name.label("model_b"),
            )
            .join(Eval.user)
            .join(ChatMessage1, Eval.message_1_id == ChatMessage1.c.message_id)
            .join(ChatMessage2, Eval.message_2_id == ChatMessage2.c.message_id)
            .join(Turn, Eval.turn_id == Turn.turn_id)
            .join(Chat, Turn.chat_id == Chat.chat_id)
            .join(
                ChatMessage,
                (ChatMessage.turn_id == Turn.turn_id) & (ChatMessage.message_type == MessageType.USER_MESSAGE),
            )
            .where(
                Eval.deleted_at.is_(None),  # type: ignore
                Eval.eval_type == EvalType.SLIDER_V0,
                Eval.score_1.is_not(None),  # type: ignore
                Chat.deleted_at.is_(None),  # type: ignore
            )
        )

        if category_names and OVERALL_CATEGORY_NAME not in category_names:
            query = query.join(Category, Category.category_id == ChatMessage.category_id)
            query = query.where(func.lower(Category.name).in_([name.lower() for name in category_names]))

        if exclude_ties:
            query = query.where(Eval.score_1 != Eval.score_2)

        if from_date is not None:
            query = query.where(Eval.created_at >= from_date)  # type: ignore
        if to_date is not None:
            query = query.where(Eval.created_at <= to_date)  # type: ignore

        if user_from_date is not None:
            query = query.where(User.created_at >= user_from_date)  # type: ignore

        if user_to_date is not None:
            query = query.where(User.created_at <= user_to_date)  # type: ignore

        if language_codes:
            enum_codes = []
            for code in language_codes:
                if enum_code := LanguageCode.from_string(code):  # type: ignore
                    enum_codes.append(enum_code)
                else:
                    raise ValueError(f"Invalid language code: {code}")
            query = query.where(ChatMessage.language_code.in_(enum_codes))  # type: ignore

        query = query.order_by(Eval.created_at)

        # Replay evals from the database.
        added = 0
        with Session(get_engine()) as session:
            results = session.exec(query).all()
            counts_by_user: dict[str, int] = defaultdict(int)
            for result_a, eval_type, user_email, model_a, model_b in results:
                if eval_type == EvalType.SLIDER_V0:
                    result_a /= 100.0
                if model_a and model_b:
                    self.update(
                        model_a=model_a,
                        model_b=model_b,
                        result_a=result_a,
                    )
                    added += 1
                    counts_by_user[user_email] += 1
                else:
                    logger.warning(f"Skipping eval with missing models: model_a={model_a}, model_b={model_b}")

            logger.info(f"Added {added} evals to the ranker. Counts per user:")
            for user, count in counts_by_user.items():
                logger.info(f"- {user}: {count}")

        return added


class ConfidenceIntervalRankerMixin(ABC):
    @abstractmethod
    def get_confidence_intervals(self) -> dict[str, ConfInterval]:
        """Returns the confidence intervals of all models."""
        raise NotImplementedError

    @abstractmethod
    def get_rating_conf_intervals(self, model: str) -> tuple[float | None, float | None, float | None]:
        """Returns the rating of `model`, along with its confidence interval."""
        raise NotImplementedError

    @abstractmethod
    def get_ratings_conf_intervals(self) -> dict[str, tuple[float, ConfInterval]]:
        """Returns the ratings of all models, along with their confidence intervals."""
        raise NotImplementedError


class EloRanker(Ranker):
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
        super().__init__(models, costs)
        self.init_rating = init_rating
        self.k = k
        self.base = base
        self.scale = scale
        # Maps models to their rating.
        self.ratings: dict[str, float] = defaultdict(lambda: self.init_rating)
        # Maps models to the list of score adjustments from the initial rating.
        self.adjustments: dict[str, list[float]] = defaultdict(list)
        self.battles = pd.DataFrame(columns=["model_a", "model_b", "result_a"])
        for battle in battles:
            self.update(battle.model_a, battle.model_b, battle.result_a)

    def update(self, model_a: str, model_b: str, result_a: float, category: str | None = None) -> None:
        """Update the ranker with a result of a battle."""
        if not (0 <= result_a <= 1):
            raise ValueError(f"Result must be between 0 and 1, got {result_a}")

        if model_a not in self.costs:
            raise ValueError(f"Model {model_a} not found; please add it first")
        if model_b not in self.costs:
            raise ValueError(f"Model {model_b} not found; please add it first")

        super().update(model_a, model_b, result_a, category)

        battle = pd.DataFrame([[model_a, model_b, result_a]], columns=self.battles.columns)
        if self.battles.empty:
            # Not strictly required, but avoids a pandas FutureWarning.
            self.battles = battle
        else:
            self.battles = pd.concat([self.battles, battle], ignore_index=True)

        rating_a = self.ratings[model_a]
        rating_b = self.ratings[model_b]
        expected_a = 1 / (1 + self.base ** ((rating_b - rating_a) / self.scale))
        expected_b = 1 / (1 + self.base ** ((rating_a - rating_b) / self.scale))

        # The amount that needs to be changed per model.
        adjustment_a = self.k * (result_a - expected_a)
        adjustment_b = self.k * (1 - result_a - expected_b)
        self.ratings[model_a] += adjustment_a
        self.ratings[model_b] += adjustment_b

        # Log the adjustments, for observability.
        self.adjustments[model_a].append(adjustment_a)
        self.adjustments[model_b].append(adjustment_b)

    def get_rating(self, model: str) -> float | None:
        if model not in self.ratings:
            self.add_model(model, DEFAULT_COST)
            self.ratings[model] = self.init_rating
        return self.ratings.get(model)

    def get_ratings(self) -> dict[str, float]:
        return self.ratings

    def get_probabilities(self) -> dict[str, float]:
        """Return the probabilities of all models."""
        return _elo_to_probabilities(self.get_ratings())

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
        rating_a = self.get_rating(model_a)
        rating_b = self.get_rating(model_b)
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
        tie_policy: Literal["ignore", "add_twice"] = "ignore",
        choix_ranker_algorithm: str = CHOIX_DEFAULT_ALGORITHM,
        choix_alpha: float = 1e-7,
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
        self.ratings: dict[str, float] = {}
        if models:
            self.model_ids = {model: i for i, model in enumerate(models)}
            self.ratings = {model: 1.0 / float(len(models)) for model in models}
        # A list of battles, where each battle is a pair of model IDs.
        self.battles: list[tuple[int, int]] = []
        self.tie_policy = tie_policy
        # Controls whether ratings should be updated.
        self.update_ratings_counter = ThresholdCounter()
        # The parameters for the underlying Choix ranker.
        self.choix_params: list[float] = []
        self.choix_ranker = getattr(choix, choix_ranker_algorithm)
        self.choix_alpha = choix_alpha
        self.choix_kwargs = choix_kwargs or {}
        for battle in battles:
            self.update(battle.model_a, battle.model_b, battle.result_a)

    def get_num_samples_no_ties(self, model: str) -> int:
        """Return the number of samples for `model`."""
        return self.wins.get(model, 0) + self.losses.get(model, 0)

    def get_num_samples(self, model: str) -> int:
        """Return the number of samples for `model`."""
        return self.get_num_samples_no_ties(model) + self.ties.get(model, 0)

    def add_model(self, model: str, cost: float = DEFAULT_COST) -> None:
        """Add a model to the ranker."""
        super().add_model(model, cost)
        self.model_ids[model] = len(self.model_ids)

    def update(self, model_a: str, model_b: str, result_a: float, category: str | None = None) -> None:
        """Update the ranker with a result of a battle."""
        super().update(model_a, model_b, result_a, category)
        self.update_ratings_counter.increment()
        # Assign new model IDs if needed.
        for model in [model_a, model_b]:
            if model not in self.model_ids:
                self.add_model(model, DEFAULT_COST)

        model_a_id = self.model_ids[model_a]
        model_b_id = self.model_ids[model_b]
        if result_a > 0.5:
            self.battles.append((model_a_id, model_b_id))
        elif result_a < 0.5:
            self.battles.append((model_b_id, model_a_id))
        else:
            if self.tie_policy == "add_twice":
                self.battles.append((model_a_id, model_b_id))
                self.battles.append((model_b_id, model_a_id))

    def get_rating(self, model: str) -> float | None:
        """Returns the rating of `model`."""
        self._maybe_update_ratings()
        return self.ratings.get(model)

    def get_ratings(self) -> dict[str, float]:
        """Returns the ratings of all models."""
        self._maybe_update_ratings()
        return self.ratings

    def get_probabilities(self) -> dict[str, float]:
        """Return the probabilities of all models."""
        return _elo_to_probabilities(self.get_ratings())

    def annotate_model(self, model: str) -> str:
        """Returns an annotation for `model`."""
        return f"Wins: {self.wins[model]}, Losses: {self.losses[model]}, Ties: {self.ties[model]}"

    def convert_to_elo_scale(self) -> None:
        """Convert the ratings to the Elo range."""
        self.ratings = {model: _score_to_elo(rank) for model, rank in self.ratings.items()}

    def _maybe_update_ratings(self) -> None:
        """Recalculate ratings, if they are stale."""
        if not self.update_ratings_counter.is_threshold_reached():
            return
        self.update_ratings()

    def update_ratings(self) -> None:
        self.update_ratings_counter.reset()

        ids_to_models = {id: model for model, id in self.model_ids.items()}
        try:
            self.choix_params = self.choix_ranker(
                len(self.model_ids), self.battles, alpha=self.choix_alpha, **self.choix_kwargs
            )
            self.ratings = {ids_to_models[id]: rank for id, rank in enumerate(self.choix_params)}
            self.convert_to_elo_scale()

        except IndexError:
            # Choix can fail when there are too few battles.
            logger.warning("Choix failed; setting all ratings to 0")
            self.ratings = {model: 0 for model in self.model_ids.keys()}

    def leaderboard(self) -> list[RatedModel]:
        self._maybe_update_ratings()
        return super().leaderboard()

    def predict(self, model_a: str, model_b: str) -> float | None:
        self._maybe_update_ratings()
        model_a_id = self.model_ids.get(model_a)
        model_b_id = self.model_ids.get(model_b)
        if model_a_id is None or model_b_id is None:
            raise ValueError(f"Model '{model_a}' or '{model_b}' not found")
        prob_a_wins, _ = choix.probabilities([model_a_id, model_b_id], self.choix_params)
        return float(prob_a_wins)

    def annotate_prediction(self, model_a: str, model_b: str) -> str:
        ratings = self.get_ratings()
        rating_a = ratings[model_a]
        rating_b = ratings[model_b]
        return f"{rating_a=:.0f}, {rating_b=:.0f}"


class ChoixRankerConfIntervals(ChoixRanker, ConfidenceIntervalRankerMixin):
    """A ChoixRanker that adds confidence intervals to the predictions."""

    ranker_type = "choix_conf_intervals"

    def __init__(
        self,
        models: list[Any] | None = None,
        costs: list[float] | None = None,
        battles: Iterable[Battle] = (),
        tie_policy: Literal["ignore", "add_twice"] = "ignore",
        choix_ranker_algorithm: str = CHOIX_DEFAULT_ALGORITHM,
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

    def get_rating_conf_intervals(self, model: str) -> tuple[float | None, float | None, float | None]:
        """Returns the rating of `model`, along with its confidence interval."""
        self._maybe_update_ratings()
        rating = self.get_rating(model)
        conf_intervals = self.conf_intervals.get(model, (None, None))
        return rating, conf_intervals[0], conf_intervals[1]

    def get_confidence_intervals(self) -> dict[str, ConfInterval]:
        """Returns the confidence intervals of all models."""
        self._maybe_update_ratings()
        return self.conf_intervals

    def convert_to_elo_scale(self) -> None:
        super().convert_to_elo_scale()
        self.conf_intervals = {
            model: (_score_to_elo(lower), _score_to_elo(upper)) for model, (lower, upper) in self.conf_intervals.items()
        }

    def get_ratings_conf_intervals(self) -> dict[str, tuple[float, ConfInterval]]:
        """Returns the ratings of all models, along with their confidence intervals."""
        return {model: (rank, self.conf_intervals[model]) for model, rank in self.get_ratings().items()}

    def annotate_model(self, model: str) -> str:
        """Returns an annotation for `model`."""
        annotation = super().annotate_model(model)
        if model in self.conf_intervals:
            annotation += f" ({self.conf_intervals[model][0]:.1f} to {self.conf_intervals[model][1]:.1f})"
        return annotation

    def _maybe_update_ratings(self) -> None:
        """Recalculate ratings, if they are stale."""
        if not self.update_ratings_counter.is_threshold_reached():
            return
        self.update_ratings()

    def get_rated_model(self, model: str) -> RatedModel:
        rated_model = super().get_rated_model(model)
        conf_interval = self.get_confidence_intervals().get(model)
        if conf_interval is not None:
            rated_model.rating_lower = conf_interval[0]
            rated_model.rating_upper = conf_interval[1]
        return rated_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.1),
        after=after_log(logger, logging.WARNING),
        retry=retry_if_exception_type((OperationalError, DatabaseError)),
    )
    def to_db(self, category_name: str | None = None, snapshot_timestamp: datetime | None = None) -> None:
        if not self.battles:
            logger.warning(f"No battles to rank for category '{category_name}'")
            return
        """Save the ratings to the database."""
        if not category_name:
            category_name = OVERALL_CATEGORY_NAME
        if not snapshot_timestamp:
            snapshot_timestamp = datetime.now()
        with Session(get_engine()) as session:
            category = session.exec(select(Category).where(Category.name == category_name)).first()
            if not category:
                raise ValueError(f"Category '{category_name}' not found")
            # Make sure the ratings are up to date.
            self.update_ratings()
            category_id = category.category_id
            llms = session.exec(select(LanguageModel)).all()
            llm_name_to_id = {llm.internal_name: llm.language_model_id for llm in llms}
            # This stores the new RatingHistory objects we create, so that we can link them to the Rating objects later.
            llm_ids_to_ranking_history = {}
            for model_name, (score, conf_interval) in self.get_ratings_conf_intervals().items():
                model_id = llm_name_to_id.get(model_name)
                if not model_id:
                    logger.warning(f"Model '{model_name}' not found")
                    continue
                rating_history = RatingHistory(
                    language_model_id=model_id,
                    category_id=category_id,
                    score=score,
                    score_lower_bound_95=conf_interval[0],
                    score_upper_bound_95=conf_interval[1],
                    snapshot_timestamp=snapshot_timestamp,
                    wins=self.wins[model_name],
                    losses=self.losses[model_name],
                    ties=self.ties[model_name],
                )
                llm_ids_to_ranking_history[model_id] = rating_history
                session.add(rating_history)

                # Query for existing Rating, to change its rating_history_id if needed.
                existing_rating = session.exec(
                    select(Rating).where(Rating.language_model_id == model_id, Rating.category_id == category_id)
                ).first()

                if existing_rating:
                    existing_rating.rating_history_id = rating_history.rating_history_id
                else:
                    new_rating = Rating(
                        language_model_id=model_id,
                        category_id=category_id,
                        rating_history_id=rating_history.rating_history_id,
                    )
                    session.add(new_rating)
            session.commit()
            logger.info(f"Saved {len(llm_ids_to_ranking_history)} rating histories to the database.")

    def _get_bootstrap_ratings(self, battles: np.ndarray, num_bootstrap_samples: int) -> list[np.ndarray]:
        # There's some overhead to spinning up workers (these are processes, not threads), so vary the number of workers
        # based on the number of battles.
        num_workers = min(MAX_NUM_WORKERS, MIN_NUM_WORKERS + len(battles) // BATTLES_PER_WORKER)
        # The parameters for each bootstrap iteration.
        params = [
            (
                self.seed + i,
                len(self.model_ids),
                battles,
                self.choix_alpha,
                self.choix_ranker,
                self.choix_kwargs,
                num_bootstrap_samples,
            )
            for i in range(self.num_bootstrap_iterations)
        ]
        if num_workers == 1:
            # Just run the iterations sequentially.
            return [_bootstrap_iteration(*params[i]) for i in range(self.num_bootstrap_iterations)]
        else:
            # Run the iterations in parallel.
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(_bootstrap_iteration, *params[i]) for i in range(self.num_bootstrap_iterations)
                ]
                return [future.result() for future in concurrent.futures.as_completed(futures)]

    def update_ratings(self) -> None:
        self.update_ratings_counter.reset()

        # Use bootstrap to estimate the confidence intervals.
        ids_to_models = {id: model for model, id in self.model_ids.items()}
        bootstrap_ratings = []  # The ratings of the models after each bootstrap iteration.
        battles = np.array(self.battles)  # Convert battles to a numpy array for efficient sampling.
        num_bootstrap_samples = int(len(self.battles) * self.bootstrap_sample_fraction)
        try:
            bootstrap_ratings = self._get_bootstrap_ratings(battles, num_bootstrap_samples)
            med_ratings = np.median(bootstrap_ratings, axis=0)
            lower_bounds = np.percentile(bootstrap_ratings, 5, axis=0)
            upper_bounds = np.percentile(bootstrap_ratings, 95, axis=0)

            # The ranker uses the median as the final rank.
            self.choix_params = med_ratings
            self.ratings = {ids_to_models[id]: rating for id, rating in enumerate(med_ratings)}
            self.conf_intervals = {
                ids_to_models[id]: (lower, upper)
                for id, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds, strict=True))
            }
            self.convert_to_elo_scale()
        except IndexError:
            # Choix can fail when there are too few battles.
            logger.warning("Choix failed; setting all ratings to 0")
            self.ratings = {model: 0 for model in self.model_ids.keys()}


class PerCategoryRanker(Ranker):
    """A ranker that stores both overall ratings and per-category ratings.

    Implemented as a map between categories and individual rankers per category.
    """

    ranker_type = "per_category"

    def __init__(self, categories: tuple[str, ...], ranker_cls: type[Ranker], ranker_kwargs: dict[str, Any]):
        self.ranker_cls = ranker_cls
        self.rankers = {category: ranker_cls(**ranker_kwargs) for category in categories}
        self.overall_ranker = ranker_cls(**ranker_kwargs)

    def get_ranker(self, category: str) -> Ranker | None:
        if category == OVERALL_CATEGORY_NAME:
            return self.overall_ranker
        return self.rankers.get(category)

    def _apply_to_all_categories(self, func: Callable[..., Any], **kwargs: Any | None) -> dict[str, Any]:
        if kwargs is None:
            kwargs = {}
        all_categories = [OVERALL_CATEGORY_NAME] + list(self.rankers)
        return {category: func(self.get_ranker(category), **kwargs) for category in all_categories}

    def get_rating_all_categories(self, model: str) -> dict[str, float | None]:
        """Return the relative rank of a model, compared to others."""
        return self._apply_to_all_categories(self.ranker_cls.get_rating, model=model)

    def get_annotated_rating_all_categories(self, model: str) -> dict[str, AnnotatedFloat]:
        """Return the rating of a model, with an explanation."""
        return self._apply_to_all_categories(self.ranker_cls.get_annotated_rating, model=model)

    def get_ratings_all_categories(self) -> dict[str, dict[str, float]]:
        """Return the ratings of all models."""
        return self._apply_to_all_categories(self.ranker_cls.get_ratings)

    def annotate_model_all_categories(self, model: str) -> dict[str, str]:
        """Return an annotation for a model."""
        return self._apply_to_all_categories(self.ranker_cls.annotate_model, model=model)

    def annotate_prediction_all_categories(self, model_a: str, model_b: str) -> dict[str, str]:
        """Return an annotation for a prediction."""
        return self._apply_to_all_categories(self.ranker_cls.annotate_prediction, model_a=model_a, model_b=model_b)

    def add_model(self, model: str, cost: float = DEFAULT_COST) -> None:
        """Add a model to the ranker."""
        for ranker in self.rankers.values():
            ranker.add_model(model, cost)
        self.overall_ranker.add_model(model, cost)

    def update(self, model_a: str, model_b: str, result_a: float, category: str | None = None) -> None:
        """Update the ranker with a result of a battle."""
        if category and category != OVERALL_CATEGORY_NAME:
            category_ranker = self.get_ranker(category)
            if category_ranker:
                category_ranker.update(model_a, model_b, result_a)
            else:
                raise ValueError(f"Category '{category}' not found")
        self.overall_ranker.update(model_a, model_b, result_a)

    def get_rating(self, model: str) -> float | None:
        return self.overall_ranker.get_rating(model)

    def get_annotated_rating(self, model: str) -> AnnotatedFloat:
        return self.overall_ranker.get_annotated_rating(model)

    def get_ratings(self) -> dict[str, float]:
        return self.overall_ranker.get_ratings()

    def get_probabilities(self) -> dict[str, float]:
        return self.overall_ranker.get_probabilities()

    def get_models(self) -> set[str]:
        return self.overall_ranker.get_models()

    def annotate_model(self, model: str) -> str:
        return self.overall_ranker.annotate_model(model)

    def annotate_prediction(self, model_a: str, model_b: str) -> str:
        return self.overall_ranker.annotate_prediction(model_a, model_b)

    def leaderboard(self) -> list[RatedModel]:
        return self.overall_ranker.leaderboard()

    def leaderboard_category(self, category_name: str) -> list[RatedModel]:
        ranker = self.get_ranker(category_name)
        if ranker is None:
            raise ValueError(f"Category '{category_name}' not found")
        return ranker.leaderboard()

    def leaderboard_all_categories(self) -> dict[str, list[RatedModel]]:
        return self._apply_to_all_categories(self.ranker_cls.leaderboard)

    def update_ratings(self) -> None:
        for ranker in list(self.rankers.values()) + [self.overall_ranker]:
            if hasattr(ranker, "update_ratings"):
                ranker.update_ratings()

    def to_db(self, category_name: str | None = None, snapshot_timestamp: datetime | None = None) -> None:
        """Save the ratings to the database."""
        if category_name:
            ranker = self.get_ranker(category_name)
            if not ranker:
                raise ValueError(f"Category '{category_name}' not found")
            ranker.to_db(category_name, snapshot_timestamp)
        else:
            for category, ranker in list(self.rankers.items()) + [(OVERALL_CATEGORY_NAME, self.overall_ranker)]:
                ranker.to_db(category, snapshot_timestamp)

    def get_total_battles(self) -> int:
        """Return the total number of battles."""
        return self.overall_ranker.get_total_battles()


RANKER_CLASSES: tuple[type[Ranker], ...] = (EloRanker, ChoixRanker, ChoixRankerConfIntervals)
RANKER_TYPES: list[str] = [cls.ranker_type for cls in RANKER_CLASSES]


def _bootstrap_iteration(
    seed: int,
    num_models: int,
    battles: np.ndarray,
    alpha: float,
    choix_ranker: Callable[..., Any],
    choix_kwargs: dict[str, Any],
    num_bootstrap_samples: int,
) -> Any:
    rng = np.random.default_rng(seed)
    return choix_ranker(
        num_models,
        rng.choice(battles, size=num_bootstrap_samples, replace=True),
        alpha=alpha,
        **choix_kwargs,
    )


@cache
def get_ranker() -> PerCategoryRanker:
    """Returns a singleton instance of the main ranker."""
    categories = tuple(fetch_categories_with_descriptions_from_db().keys())
    return PerCategoryRanker(categories, ChoixRankerConfIntervals, DEFAULT_RANKER_KWARGS)


def get_default_ranker() -> ChoixRankerConfIntervals:
    """Returns the current default ranker withe the default params."""
    return ChoixRankerConfIntervals()
