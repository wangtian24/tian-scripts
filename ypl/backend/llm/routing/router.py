import heapq
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import chain
from typing import Any

import numba
import numpy as np
from google.auth import default
from google.cloud import logging as goog_logging
from google.cloud import run_v2
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.config import settings
from ypl.backend.db import get_async_engine
from ypl.backend.llm.constants import MODEL_HEURISTICS
from ypl.backend.llm.ranking import ConfidenceIntervalRankerMixin, Ranker, get_ranker
from ypl.backend.llm.routing.policy import SelectionCriteria, decayed_random_fraction
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, Provider
from ypl.utils import RNGMixin, async_timed_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RouterState(BaseModel):
    """
    The current state of the routing pipeline, used to track the models that have been selected and those that
    are excluded.
    """

    selected_models: dict[str, dict[SelectionCriteria, float]] = {}  # models to selection criterias and scores
    excluded_models: set[str] = set()  # models to exclude
    all_models: set[str] = set()  # all models to choose from

    def __add__(self, other: "RouterState") -> "RouterState":
        """
        Merge two RouterStates together, adding the scores of duplicate model-criteria pairs.
        """
        merged_selected_models: dict[str, dict[SelectionCriteria, float]] = {}

        for model, criteria_map in chain(other.selected_models.items(), self.selected_models.items()):
            if model in self.excluded_models or model in other.excluded_models:
                continue

            for criteria, score in criteria_map.items():
                if model not in merged_selected_models:
                    merged_selected_models[model] = {}
                    merged_selected_models[model][criteria] = score
                elif criteria not in merged_selected_models[model]:
                    merged_selected_models[model][criteria] = score
                else:
                    merged_selected_models[model][criteria] += score

        return RouterState(
            selected_models=merged_selected_models,
            excluded_models=self.excluded_models.union(other.excluded_models),
            all_models=self.all_models.union(other.all_models),
        )

    def deepcopy(self) -> "RouterState":
        """Return a deep copy of the RouterState."""
        return self.model_copy(deep=True)

    def get_selected_models(self) -> set[str]:
        """
        Return the models that have been selected.
        """
        return set(self.selected_models.keys())

    def get_sorted_selected_models(self) -> list[str]:
        """
        Return the models that have been selected, sorted by the sum of their scores.
        """
        return sorted(self.get_selected_models(), key=lambda x: sum(self.selected_models[x].values()), reverse=True)

    def get_selectable_models(self) -> set[str]:
        """
        Return the models that are not excluded.
        """
        return self.all_models - self.excluded_models

    def multiply_scores(self, factor: float) -> None:
        """
        Multiply the scores of all selected models by a factor.
        """
        for model, criteria_map in self.selected_models.items():
            for criteria, score in criteria_map.items():
                self.selected_models[model][criteria] = score * factor

    def get_model_score_map(self) -> dict[str, float]:
        """
        Return a map of the models to their scores, summed over all criteria.
        """
        return {model: sum(criteria_map.values()) for model, criteria_map in self.selected_models.items()}

    @classmethod
    @async_timed_cache(seconds=600)  # Cache for 10 minutes
    async def new_all_models_state(cls) -> "RouterState":
        query = (
            select(LanguageModel.internal_name)
            .distinct()
            .join(Provider)
            .where(
                LanguageModel.deleted_at.is_(None),  # type: ignore
                LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
                Provider.deleted_at.is_(None),  # type: ignore
                Provider.is_active.is_(True),  # type: ignore
            )
        )
        async with AsyncSession(get_async_engine()) as session:
            models = await session.exec(query)

        return RouterState(
            selected_models={},
            excluded_models=set(),
            all_models=set(models.all()),
        )


class RouterModule(ABC):
    """
    Abstract base class for a module in the routing pipeline. Routing is defined as passing a state through an
    arithmetic circuit of modules, which may include selection, filtering, mapping, branching, and sequential
    and parallel composition, akin to MapReduce or a Unix pipeline. A router is composed of a sequence of
    arithmetic operations on RouterModule, e.g.,

    .. code-block::python

        router = CostModelProposer() | TopK(3)
        state = router.select_models(state=RouterState(all_models=set(FRONTEND_MODELS)))
        print(state.get_selected_models())

    would route to the top 3 cheapest models out of the available front-end models, passing the state
    sequentially through CostModelProposer to TopK. These three arithmetic operations are supported:

    .. code-block::python

        # Sequential composition; the state is passed sequentially through each module
        router = CostModelProposer() | EloProposer(ranker)

        # Parallel composition; the state is split across two modules, which operate in parallel.
        # The results of each module are merged together (see :py:meth:`.RouterState.__add__`).
        router = CostModelProposer() & EloProposer(ranker)

        # Exclusive drawing; the state is randomly routed to one of the two modules, with a probability
        # given by the `with_probs` method.
        router = CostModelProposer() ^ EloProposer(ranker)

    These modules can be chained and combined arbitrarily, e.g.,

    .. code-block::python

        router = (
            (((EloProposer(ranker) & CostModelProposer()) | TopK(3))
            ^ ConfidenceIntervalWidthModelProposer(ranker)).with_probs(0.75, 0.25)
        ) | RoutingDecisionLogger()

    See Also:
    - :py:meth:`.RouterSequentialChain`, :py:meth:`.RouterParallelChain`, :py:meth:`.RouterExclusiveChain`
    - :py:meth:`.RouterState.__add__`, :py:meth:`.RouterState.multiply_scores`
    - :py:meth:`.RouterDecisionLogger`
    """

    _multiplier: float | None = None

    def select_models(self, state: RouterState | None = None) -> RouterState:
        response = self._select_models(state or RouterState())

        if self._multiplier is not None:
            response.multiply_scores(self._multiplier)

        return response

    async def aselect_models(self, state: RouterState | None = None) -> RouterState:
        response = await self._aselect_models(state or RouterState())

        if self._multiplier is not None:
            response.multiply_scores(self._multiplier)

        return response

    def with_multiplier(self, multiplier: float) -> "RouterModule":
        self._multiplier = multiplier
        return self

    def __or__(self, other: "RouterModule") -> "RouterModule":
        return RouterSequentialChain(self, other)

    def __and__(self, other: "RouterModule") -> "RouterModule":
        return RouterParallelChain(self, other)

    def __xor__(self, other: "RouterModule") -> "RouterExclusiveChain":
        return RouterExclusiveChain(self, other)

    @abstractmethod
    def _select_models(self, state: RouterState) -> RouterState:
        raise NotImplementedError

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return self._select_models(state)


class RouterDecisionLogger(RouterModule):
    def __init__(self, enabled: bool = True, prefix: str = "router") -> None:
        """
        Args:
            enabled: Whether to log the routing decision.
        """
        self.enabled = enabled
        self.prefix = prefix

    def _select_models(self, state: RouterState) -> RouterState:
        """
        Log the routing decision.
        """
        if self.enabled:
            decision = RoutingDecision(
                prefix=self.prefix,
                candidate_model_names=list(state.all_models),
                chosen_model_names=list(state.selected_models.keys()),
                selection_criteria=list(
                    dict.fromkeys(
                        [
                            str(criteria)
                            for _, criteria_map in state.selected_models.items()
                            for criteria in criteria_map.keys()
                        ]
                    )
                ),
            )
            decision.log(structured_output=True)

        return state


class RouterSequentialChain(RouterModule):
    def __init__(self, *args: RouterModule) -> None:
        """
        Represents a sequential chain of routing modules, i.e., the state is passed sequentially through each module.
        See :py:class:`.RouterModule` for more information.

        Args:
            *args: The modules to pass the state through sequentially.
        """
        self.router_modules: list[RouterModule] = list(args)

    def __or__(self, other: "RouterModule") -> "RouterModule":
        self.router_modules.append(other)
        return self

    def _select_models(self, state: RouterState) -> RouterState:
        for router_module in self.router_modules:
            state = router_module.select_models(state=state)

        return state


class RouterExclusiveChain(RNGMixin, RouterModule):
    def __init__(self, *args: RouterModule) -> None:
        """
        Represents an exclusive chain of routing modules, i.e., the state is randomly routed to one of the modules.
        See :py:class:`.RouterModule` for more information.

        Args:
            *args: The modules to randomly route the state to.
        """
        self.router_modules: list[RouterModule] = list(args)
        self.random_probabilities: list[float] = []

    def __xor__(self, other: "RouterModule") -> "RouterExclusiveChain":
        self.router_modules.append(other)
        return self

    def with_probs(self, *probabilities: float) -> "RouterExclusiveChain":
        """
        Set the probabilities of routing to each module in the same order that it was constructed.

        Args:
            *probabilities: The probabilities of routing to each module.
        """
        p = np.array(probabilities)
        p = p / p.sum()
        self.random_probabilities = p.tolist()

        return self

    def _select_models(self, state: RouterState) -> RouterState:
        if len(self.random_probabilities) != len(self.router_modules):
            logging.warning(
                "Random probabilities not set for RouterExclusiveChain; using default of 1/len(router_modules)"
            )
            probs = np.full(len(self.router_modules), 1 / len(self.router_modules))
        else:
            probs = np.array(self.random_probabilities)

        chosen_module = self.get_rng().choice(np.array(self.router_modules, dtype=object), replace=False, p=probs)
        state = chosen_module.select_models(state=state)

        return state


class RouterParallelChain(RouterModule):
    def __init__(self, *args: RouterModule) -> None:
        """
        Represents a parallel chain of routing modules, i.e., the state is split across all modules.
        See :py:class:`.RouterModule` for more information.

        Args:
            *args: The modules to pass the state through in parallel.
        """
        self.router_modules: list[RouterModule] = list(args)

    def __and__(self, other: "RouterModule") -> "RouterModule":
        self.router_modules.append(other)
        return self

    def _select_models(self, state: RouterState) -> RouterState:
        responses = []

        for router_module in self.router_modules:
            router_response = router_module.select_models(state=state.deepcopy())
            responses.append(router_response)

        for response in responses:
            state += response

        return state


class ModelProposer(RouterModule):
    """
    Represents a proposer of a set of models to route to. Subclasses should implement the `_propose_models` method
    to define the specific model selection logic.
    """

    def _select_models(self, state: RouterState) -> RouterState:
        """
        Propose a set of models to route to.

        Args:
            num_models: The number of models to propose.
            state: The state to propose models for. The `all_models` field is used to determine the set of
                models to propose, e.g., `FRONTEND_MODELS`.
        """
        models_to_select = state.get_selectable_models()
        response = self._propose_models(models_to_select, state.deepcopy())

        return response

    async def _aselect_models(self, state: RouterState) -> RouterState:
        models_to_select = state.get_selectable_models()
        response = await self._apropose_models(models_to_select, state.deepcopy())

        return response

    @abstractmethod
    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        """
        Propose a set of models to route to.

        Args:
            num_models: The number of models to propose.
            models_to_select: The set of models to propose.
            state: The state to propose models for.
        """
        raise NotImplementedError

    async def _apropose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        return self._propose_models(models_to_select, state)


class RandomModelProposer(RNGMixin, ModelProposer):
    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        """
        Randomly select a set of models to route to.

        Args:
            num_models: The number of models to propose.
            models_to_select: The set of models to randomly select from.
            state: The state to propose models for.
        """
        selected_models = self.get_rng().choice(list(models_to_select), len(models_to_select), replace=False)

        return RouterState(
            selected_models={model: {SelectionCriteria.RANDOM: self.get_rng().random()} for model in selected_models},
            excluded_models=state.excluded_models,
        )


class ModelFilter(RouterModule):
    """
    Represents a filter of a set of models to route to. Subclasses should implement the `_filter` method
    to define the specific model filtering logic.
    """

    def __init__(self, persist: bool = False) -> None:
        """
        Args:
            persist: Whether to persist the models that are excluded from the selection, so that they are never
                selected again in future routing modules.
        """
        self.persist = persist

    def _select_models(self, state: RouterState) -> RouterState:
        state, excluded_models = self._filter(state)

        if self.persist:
            state.excluded_models = state.excluded_models.union(excluded_models)

        return state

    @abstractmethod
    def _filter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        """
        Filter a set of models to route to.

        Args:
            state: The state to filter models for.
        """
        raise NotImplementedError


class TopK(ModelFilter):
    """
    Represents a filter of a set of models to route to. The top-k models are selected, where the score is the sum of the
    scores of the models.
    """

    def __init__(self, k: int, persist: bool = True) -> None:
        """
        Args:
            k: The number of top-k models to select.
        """
        super().__init__(persist=persist)
        self.k = k

    def _filter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        selected_models = sorted(state.selected_models.items(), key=lambda x: sum(x[1].values()), reverse=True)
        excluded_models = {model for model, _ in selected_models[self.k :]}

        state.selected_models = {model: x for model, x in state.selected_models.items() if model not in excluded_models}

        return state, excluded_models


class MinimumFractionModelProposer(RNGMixin, ModelProposer):
    def __init__(
        self,
        num_models: int,
        minimum_model_traffic_fraction: dict[str, float],
    ) -> None:
        self.minimum_model_traffic_fraction = minimum_model_traffic_fraction
        self.num_models = num_models

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        if self.num_models > len(models_to_select):
            return state

        models = list(models_to_select)
        probs = np.array([self.minimum_model_traffic_fraction.get(model, 0) for model in models])
        probs /= probs.sum()

        selected_models = self.get_rng().choice(models, min(self.num_models, len(models)), replace=False, p=probs)

        return RouterState(
            selected_models={model: {SelectionCriteria._MIN_TRAFFIC_FRACTION: 1.0} for model in selected_models},
            excluded_models=state.excluded_models,
        )


class RandomShuffle(RNGMixin, ModelProposer):
    def __init__(self, shuffle_same_scores_only: bool = False) -> None:
        self.shuffle_same_scores_only = shuffle_same_scores_only

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        if self.shuffle_same_scores_only:
            # Shuffle only the order of models with the same score
            model_score_map = state.get_model_score_map()
            score_model_map: dict[float, list[str]] = {score: [] for score in model_score_map.values()}
            shuffled_items = list(model_score_map.items())
            self.get_rng().shuffle(shuffled_items)  # type: ignore

            for model, score in shuffled_items:
                score_model_map[score].append(model)

            selected_models = {}

            for score, models in score_model_map.items():
                for model in models:
                    selected_models[model] = {SelectionCriteria.RANDOM: score}

            return RouterState(
                selected_models=selected_models, excluded_models=state.excluded_models, all_models=set(models_to_select)
            )
        else:
            selected_models = self.get_rng().choice(list(models_to_select), len(models_to_select), replace=False)  # type: ignore

            return RouterState(
                selected_models={model: {SelectionCriteria.RANDOM: 1.0} for model in selected_models},
                excluded_models=state.excluded_models,
                all_models=set(models_to_select),
            )


class AlwaysGoodModelMetaRouter(ModelProposer):
    """
    A meta-router that ensures that at least one "good" model is always selected, where the definition of "good" is
    defined by the `num_good` parameter and the `ranker` parameter, which is a :py:class:`.Ranker` used to rank
    the models.
    """

    def __init__(self, ranker: Ranker, router: RouterModule, num_good: int) -> None:
        self.router = router
        self.num_good = num_good
        self.ranker = ranker

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        good_model_response = (EloProposer(self.ranker) | TopK(self.num_good)).select_models(state.deepcopy())
        good_model_response.all_models = good_model_response.get_selected_models()
        good_model_response = self.router.select_models(good_model_response)

        top1_filter = TopK(1)
        good_model_response = top1_filter.select_models(state=good_model_response)
        good_model_response.multiply_scores(1000)  # ensures that one good model is always selected
        state.excluded_models.update(good_model_response.selected_models.keys())

        default_response = self.router.select_models(state.deepcopy())
        default_response.excluded_models = set()
        good_model_response.excluded_models = set()
        good_model_response += default_response

        return good_model_response


class EloProposer(RNGMixin, ModelProposer):
    """
    A proposer of a set of models to route to. The models are ranked by the `ranker` parameter, which is a
    :py:class:`.Ranker`.
    """

    def __init__(self, ranker: Ranker) -> None:
        """
        Args:
            ranker: The ranker to use to rank the models.
        """
        self.ranker = ranker

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        elo_ratings = self.ranker.get_ratings()

        selected_models = sorted(
            [model for model in models_to_select],
            key=lambda m: elo_ratings.get(m, 1.0) + self.get_rng().random() * 1e-7,  # add a small random jitter
            reverse=True,
        )

        max_elo = max(elo_ratings.values(), default=1.0)

        return RouterState(
            selected_models={
                model: {SelectionCriteria.TOP: elo_ratings.get(model, 1.0) / max_elo} for model in selected_models
            },
            excluded_models=state.excluded_models,
        )


class ProportionalModelProposer(RNGMixin, ModelProposer):
    """
    A proposer of a set of models to route to. The models are selected proportionally following
    :py:meth:`.Ranker.get_probabilities`.
    """

    def __init__(self, num_models: int, ranker: Ranker) -> None:
        """
        Args:
            num_models: The number of models to propose.
            ranker: The ranker to use to rank the models.
        """
        self.ranker = ranker
        self.num_models = num_models

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        probabilities = self.ranker.get_probabilities().copy()

        if not probabilities:
            return RouterState()

        for k, _ in list(probabilities.items()):
            if k not in models_to_select:
                del probabilities[k]

        if not probabilities:
            return RouterState()

        # Re-normalize probabilities to sum to 1
        probabilities = {k: v / sum(probabilities.values()) for k, v in probabilities.items()}
        models = list(probabilities.keys())

        chosen_models = set(
            self.get_rng().choice(
                models,
                size=min(self.num_models, len(models)),
                p=list(probabilities.values()),
                replace=False,
            )
        )

        return RouterState(
            selected_models={model: {SelectionCriteria.PROPORTIONAL: 1.0} for model in chosen_models},
            excluded_models=state.excluded_models,
            all_models=set(models),
        )


class ConfidenceIntervalWidthModelProposer(ModelProposer):
    """
    A proposer of a set of models to route to. The models are selected based on the width of the confidence
    intervals, where the width is defined by the `ranker`.
    """

    def __init__(self, num_models: int, ranker: ConfidenceIntervalRankerMixin) -> None:
        """
        Args:
            num_models: The number of models to propose.
            ranker: The ranker to use to rank the models.
        """
        self.ranker = ranker
        self.num_models = num_models

        if not hasattr(self.ranker, "get_confidence_intervals"):
            raise ValueError("Ranker must implement `get_confidence_intervals`")

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        conf_interval_widths = {model: abs(x[1] - x[0]) for model, x in self.ranker.get_confidence_intervals().items()}

        for model in list(conf_interval_widths.keys()):
            if model not in models_to_select:
                del conf_interval_widths[model]

        sorted_models = sorted(conf_interval_widths.items(), key=lambda x: x[1], reverse=True)
        selected_models = [model for model, _ in sorted_models[: self.num_models]]

        return RouterState(
            selected_models={
                model: {SelectionCriteria.CONF_INTERVAL_WIDTH: 1.0 / (rank + 1)}
                for rank, model in enumerate(selected_models)
            },
            excluded_models=state.excluded_models,
            all_models=set(conf_interval_widths.keys()),
        )


class ConfidenceIntervalNumOverlapModelProposer(ModelProposer):
    """
    A proposer of a set of models to route to. The models are selected based on the number of overlaps in the confidence
    intervals, where the confidence intervals are computed by the `ranker`.
    """

    def __init__(self, num_models: int, ranker: Ranker) -> None:
        """
        Args:
            ranker: The ranker to use to rank the models.
        """
        self.ranker = ranker
        self.num_models = num_models

        if not hasattr(self.ranker, "get_confidence_intervals"):
            raise ValueError("Ranker must implement `get_confidence_intervals`")

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        conf_intervals = []
        models = []

        for model, ci in self.ranker.get_confidence_intervals().items():  # type: ignore
            if model not in models_to_select:
                continue

            conf_intervals.append(ci)
            models.append(model)

        if not conf_intervals:
            return RouterState()

        num_overlaps, perm_map = _fast_compute_all_num_intersections(np.array(conf_intervals))
        k = min(self.num_models, len(models))
        sorted_ind = np.argpartition(num_overlaps, -k)[-k:]
        sorted_models = [(num_overlaps[i], models[perm_map[i]]) for i in sorted_ind]
        sorted_models = sorted(sorted_models, key=lambda x: x[0], reverse=True)

        selected_models = [model for _, model in sorted_models][:k]

        return RouterState(
            selected_models={
                model: {SelectionCriteria.CONF_INTERVAL_NUM_OVERLAP: 1.0 / (rank + 1)}
                for rank, model in enumerate(selected_models)
            },
            excluded_models=state.excluded_models,
            all_models=set(models),
        )


class ConfidenceIntervalWidthOverlapModelProposer(ModelProposer):
    """
    A proposer of a set of models to route to. The models are selected based on the overlap width of the
    confidence intervals, where the confidence intervals are computed by the `ranker`.
    """

    def __init__(self, num_models: int, ranker: Ranker) -> None:
        """
        Args:
            num_models: The number of models to propose.
            ranker: The ranker to use to rank the models.
        """
        self.num_models = num_models
        self.ranker = ranker

        if not hasattr(self.ranker, "get_confidence_intervals"):
            raise ValueError("Ranker must implement `get_confidence_intervals`")

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        conf_intervals = []
        models = []

        for model, ci in self.ranker.get_confidence_intervals().items():  # type: ignore
            if model not in models_to_select:
                continue

            conf_intervals.append(ci)
            models.append(model)

        num_models = min(self.num_models, len(models))

        if not conf_intervals or not num_models:
            return RouterState()

        highest_overlapping_pairs, _ = _fast_compute_all_conf_overlap_diffs(np.array(conf_intervals), num_models)
        sorted_ind = list(dict.fromkeys(highest_overlapping_pairs.flatten()))[:num_models]

        return RouterState(
            selected_models={
                models[i]: {SelectionCriteria.CONF_INTERVAL_PAIR_OVERLAP: 1.0 / (rank + 1)}
                for rank, i in enumerate(sorted_ind)
            },
            excluded_models=state.excluded_models,
            all_models=set(models),
        )


class CostModelProposer(ModelProposer):
    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        model_costs = []

        for model in models_to_select:
            m = model if model in MODEL_HEURISTICS else "gpt-4o-mini"
            cost = (
                MODEL_HEURISTICS[m].dollars_per_million_output_tokens
                + MODEL_HEURISTICS[m].dollars_per_million_input_tokens
            )
            model_costs.append((cost, model))

        selected_models = sorted(model_costs, key=lambda x: x[0])

        return RouterState(
            selected_models={
                model: {SelectionCriteria.MIN_SIMPLE_COST: 1 / (cost + 1)} for cost, model in selected_models
            },
            excluded_models=state.excluded_models,
            all_models=set([x[1] for x in model_costs]),
        )


@numba.njit
def _fast_compute_all_conf_overlap_diffs(intervals: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Quickly computes the difference between the confidence intervals of all pairs of intervals. Returns
    the top k interval pairs with the largest overlap. This isn't the most efficient way to do this, but practically
    it should be good enough.

    Args:
        intervals: A 2D numpy array of shape (n, 2) where each row is an interval [start, end].
        k: The number of intervals to return.

    Returns:
        A tuple of an array of length (n, 2) containing the top-k pairs and an array of the diffs
    """
    interval_heap = [(-1.0, 0, 1)]  # for Numba to successfully infer the type
    k = min((len(intervals) * (len(intervals) - 1)) // 2, k)  # number of unique pairs
    ret_inds = np.empty((k, 2), dtype=np.int32)
    ret_vals = np.empty(k, dtype=np.float64)

    for idx1 in range(len(intervals)):  # SIMD should take care of this
        for idx2 in range(idx1 + 1, len(intervals)):  # this as well
            start1, end1 = intervals[idx1]
            start2, end2 = intervals[idx2]
            overlap = max(0, min(end1, end2) - max(start1, start2))
            heapq.heappush(interval_heap, (overlap, idx1, idx2))

            if len(interval_heap) > k:
                heapq.heappop(interval_heap)

    idx = 0

    while interval_heap:
        overlap, idx1, idx2 = heapq.heappop(interval_heap)
        ret_inds[len(ret_inds) - idx - 1][0] = idx1  # better than list/tuple assignment because of Numba
        ret_inds[len(ret_inds) - idx - 1][1] = idx2
        ret_vals[len(ret_vals) - idx - 1] = overlap
        idx += 1

    return ret_inds, ret_vals


@numba.njit
def _fast_compute_all_num_intersections(intervals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Quickly computes the number of intersections between all pairs of intervals.

    Args:
        intervals: A 2D numpy array of shape (n, 2) where each row is an interval [start, end].

    Returns:
        A tuple of an array of length n containing the number of intersections per interval and a permutation
        to sort the array by start.
    """
    counts = np.zeros(len(intervals), dtype=np.int32)
    sort_idxs = intervals[:, 0].argsort()
    permutation_map = np.empty_like(sort_idxs)
    permutation_map[sort_idxs] = np.arange(len(sort_idxs))
    intervals = intervals[sort_idxs]
    interval_heap = [(-1000000.0, 0)]  # for Numba to successfully infer the type

    for idx, interval in enumerate(intervals):
        start, end = interval

        while interval_heap and interval_heap[0][0] < start:
            heapq.heappop(interval_heap)

        for x in interval_heap:
            counts[x[1]] += 1  # SIMD should take care of (parts of) this

        counts[idx] += len(interval_heap)
        heapq.heappush(interval_heap, (end, idx))

    return counts, permutation_map


class MaxSpeedProposer(ModelProposer):
    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        model_speeds = [
            (model_name, MODEL_HEURISTICS.get(model_name, MODEL_HEURISTICS["gpt-4-turbo"]).tokens_per_second)
            for model_name in models_to_select
        ]

        max_speed = max(speed for _, speed in model_speeds)

        return RouterState(
            selected_models={
                model_name: {SelectionCriteria.MAX_SPEED: speed / max_speed} for model_name, speed in model_speeds
            },
            all_models=models_to_select,
        )


class RoutingDecision:
    def __init__(
        self,
        prefix: str,
        candidate_model_names: Sequence[str],
        chosen_model_names: Sequence[str],
        selection_criteria: Sequence[str],
        **kwargs: Any,
    ) -> None:
        """
        Args:
            prefix: A prefix to identify the router that made the decision.
            candidate_model_names: The names of the models that were proposed.
            chosen_model_names: The names of the models that were finally chosen. Should be a subset of the above.
            selection_criteria: The names of the criteria that made the decision. If there are multiple, it means that
                multiple were involved in building the chosen model list, and they are inseparable.
            **kwargs: Additional metadata to log.
        """
        from ypl import __version__

        self.codebase_version = __version__
        self.prefix = prefix
        self.candidate_model_names = candidate_model_names
        self.chosen_model_names = chosen_model_names
        self.selection_criteria = selection_criteria
        self.additional_metadata = kwargs

    def log(self, structured_output: bool = False) -> None:
        """
        Logs the routing decision with a lot of schema flexibility. Parts of this should eventually be standardized
        once our router models are more concrete. The `additional_metadata` kwargs will be stored under the
        `additional_metadata` field in the log.

        Args:
            structured_output: Whether to log the decision in a structured format. This requires google-cloud-logging.
        """
        payload = {
            "codebase_version": self.codebase_version,
            "prefix": self.prefix,
            "candidate_model_names": self.candidate_model_names,
            "chosen_model_names": self.chosen_model_names,
            "selection_criteria": self.selection_criteria,
            "additional_metadata": self.additional_metadata,
        }

        if structured_output and settings.USE_GOOGLE_CLOUD_LOGGING:
            logging_client = goog_logging.Client()
            logger = logging_client.logger("routing-decisions")
            logger.log_struct(payload)
        else:
            logging.info(f"Routing decision: {payload}")


def get_router_ranker(ranker: Ranker | None = None) -> tuple[RouterModule, Ranker]:
    min_weight = settings.ROUTING_WEIGHTS.get("min_simple_cost", 0.1)
    rand_weight = settings.ROUTING_WEIGHTS.get("random", 0.1)
    top_weight = settings.ROUTING_WEIGHTS.get("top", 0.1)
    ranker = ranker or get_ranker()

    router: RouterModule = (CostModelProposer() ^ RandomModelProposer() ^ EloProposer(ranker)).with_probs(
        min_weight,
        rand_weight + decayed_random_fraction(ranker, initial_value=0.6, final_value=0.05, steps=50000),
        top_weight,
    )

    if settings.ROUTING_GOOD_MODELS_ALWAYS:
        router = AlwaysGoodModelMetaRouter(ranker, router, num_good=settings.ROUTING_GOOD_MODELS_RANK_THRESHOLD)

    router = router | TopK(2) | RouterDecisionLogger(enabled=settings.ROUTING_DO_LOGGING, prefix="default-router")

    return router, ranker


def get_router(ranker: Ranker | None = None) -> RouterModule:
    return get_router_ranker(ranker)[0]


def get_gcp_cloud_run_uri(service_name: str, region: str) -> str:
    credentials, project_id = default()
    client = run_v2.ServicesClient(credentials=credentials)
    name = f"projects/{project_id}/locations/{region}/services/{service_name}"
    request = run_v2.GetServiceRequest(name=name)
    response = client.get_service(request=request)

    return response.uri


def get_prompt_conditional_router(prompt: str, num_models: int) -> RouterModule:
    from ypl.backend.llm.routing.prompt_router import RemotePromptCategorizerProposer

    rand_weight = settings.ROUTING_WEIGHTS.get("random", 0.1)
    prompt_weight = settings.ROUTING_WEIGHTS.get("best_prompt_quality", 0.5)
    endpoint = settings.PYTORCH_SERVE_GCP_URL
    key = settings.X_API_KEY

    router: RouterModule = (
        (
            RandomModelProposer() ^ (RemotePromptCategorizerProposer(prompt, endpoint, key) | MaxSpeedProposer())
        ).with_probs(rand_weight, prompt_weight)
        | TopK(num_models)
        | RouterDecisionLogger(enabled=settings.ROUTING_DO_LOGGING, prefix="prompt-conditional-router")
    )

    return router
