import heapq
from abc import abstractmethod

import numba
import numpy as np
from ypl.backend.llm.constants import MODEL_HEURISTICS
from ypl.backend.llm.db_helpers import (
    adeduce_original_provider,
    deduce_original_providers,
    get_all_pro_models,
    get_all_strong_models,
    get_image_attachment_models,
)
from ypl.backend.llm.ranking import ConfidenceIntervalRankerMixin, Ranker
from ypl.backend.llm.routing.modules.base import RouterModule
from ypl.backend.llm.routing.modules.filters import TopK
from ypl.backend.llm.routing.policy import SelectionCriteria
from ypl.backend.llm.routing.router_state import RouterState
from ypl.utils import RNGMixin


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

        self._update_debug(response)
        return response

    async def _aselect_models(self, state: RouterState) -> RouterState:
        models_to_select = state.get_selectable_models()
        response = await self._apropose_models(models_to_select, state.deepcopy())
        self._update_debug(response)

        return response

    def _update_debug(self, response: RouterState) -> RouterState:
        for model in response.selected_models:
            criteria_names = [f"{cr.name.lower()}^{w:.2f}" for cr, w in response.selected_models[model].items()]
            if model not in response.model_journey:
                response.model_journey[model] = ""
            response.model_journey[model] += f" +{','.join(criteria_names)}"

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


class ProModelProposer(RNGMixin, ModelProposer):
    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        if not models_to_select:
            return RouterState()

        all_pro_models = self._get_all_pro_models()
        models_to_select = models_to_select.intersection(all_pro_models)
        models_to_select_ = list(models_to_select)
        self.get_rng().shuffle(models_to_select_)

        return state.emplaced(
            selected_models={model: {SelectionCriteria.PRO_MODELS: 1.0} for model in models_to_select_},
            all_models=state.all_models,
            excluded_models=state.excluded_models | (state.all_models - set(models_to_select)),
        )

    async def _apropose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        return self._propose_models(models_to_select, state)

    def _get_all_pro_models(self) -> set[str]:
        return set(get_all_pro_models())


class ImageProModelProposer(ProModelProposer):
    def _get_all_pro_models(self) -> set[str]:
        return set(get_image_attachment_models())


class StrongModelProposer(RNGMixin, ModelProposer):
    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        if not models_to_select:
            return RouterState()

        all_strong_models = set(get_all_strong_models())
        models_to_select = models_to_select.intersection(all_strong_models)
        models_to_select_ = list(models_to_select)
        self.get_rng().shuffle(models_to_select_)

        return state.emplaced(
            selected_models={model: {SelectionCriteria.STRONG_MODELS: 1.0} for model in models_to_select_},
            all_models=state.all_models,
            excluded_models=state.excluded_models | (state.all_models - set(models_to_select)),
        )

    async def _apropose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        return self._propose_models(models_to_select, state)


class RandomModelProposer(RNGMixin, ModelProposer):
    def __init__(
        self,
        *,
        for_criteria: SelectionCriteria = SelectionCriteria.RANDOM,
        models: set[str] | None = None,
        providers: set[str] | None = None,
    ) -> None:
        self.for_criteria = for_criteria
        self.models = models or set()
        self.providers = providers or set()

    def _random_select(self, models_to_select: set[str], state: RouterState) -> RouterState:
        if not models_to_select:
            return RouterState()

        selected_models = self.get_rng().choice(list(models_to_select), len(models_to_select), replace=False)

        return state.emplaced(
            selected_models={model: {self.for_criteria: self.get_rng().random()} for model in selected_models},
            all_models=state.all_models,
            excluded_models=state.excluded_models | (state.all_models - set(selected_models)),
        )

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        """
        Randomly select a set of models to route to.

        Args:
            num_models: The number of models to propose.
            models_to_select: The set of models to randomly select from.
            state: The state to propose models for.
        """
        if self.models:
            models_to_select = models_to_select.intersection(self.models)

        if self.providers:
            provider_map = deduce_original_providers(tuple(models_to_select))
            models_to_select = {model for model in models_to_select if provider_map[model] in self.providers}

        return self._random_select(models_to_select, state)

    async def _apropose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        if self.models:
            models_to_select = models_to_select.intersection(self.models)

        if self.providers:
            models_to_select = {
                model for model in models_to_select if (await adeduce_original_provider(model)) in self.providers
            }

        return self._random_select(models_to_select, state)


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

        return state.emplaced(
            selected_models={model: {SelectionCriteria._MIN_TRAFFIC_FRACTION: 1.0} for model in selected_models},
            excluded_models=state.excluded_models,
        )


# TODO(tian) - this is not used anywhere
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
                    selected_models[model] = {SelectionCriteria.RANDOM_SHUFFLE: score}

            return state.emplaced(
                selected_models=selected_models, excluded_models=state.excluded_models, all_models=set(models_to_select)
            )
        else:
            selected_models = self.get_rng().choice(list(models_to_select), len(models_to_select), replace=False)  # type: ignore

            return state.emplaced(
                selected_models={model: {SelectionCriteria.RANDOM_SHUFFLE: 1.0} for model in selected_models},
                excluded_models=state.excluded_models,
                all_models=set(models_to_select),
            )


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

        return state.emplaced(
            selected_models={
                model: {SelectionCriteria.TOP_ELO: elo_ratings.get(model, 1.0) / max_elo} for model in selected_models
            }
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

        good_model_response.model_journey = {
            model: f"+always_good({value})" for model, value in good_model_response.model_journey.items()
        }

        return good_model_response

    def _update_debug(self, response: RouterState) -> RouterState:
        # override the parent one, we are doing some customized debug above already
        return response


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

        return state.emplaced(
            selected_models={model: {SelectionCriteria.PROPORTIONAL: 1.0} for model in chosen_models},
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

        return state.emplaced(
            selected_models={
                model: {SelectionCriteria.CONF_INTERVAL_WIDTH: 1.0 / (rank + 1)}
                for rank, model in enumerate(selected_models)
            },
            all_models=set(conf_interval_widths.keys()),
        )


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

        return state.emplaced(
            selected_models={
                model: {SelectionCriteria.CONF_INTERVAL_NUM_OVERLAP: 1.0 / (rank + 1)}
                for rank, model in enumerate(selected_models)
            },
            all_models=set(models),
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

        return state.emplaced(
            selected_models={
                models[i]: {SelectionCriteria.CONF_INTERVAL_PAIR_OVERLAP: 1.0 / (rank + 1)}
                for rank, i in enumerate(sorted_ind)
            },
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

        return state.emplaced(
            selected_models={
                model: {SelectionCriteria.MIN_SIMPLE_COST: 1 / (cost + 1)} for cost, model in selected_models
            },
            all_models=set([x[1] for x in model_costs]),
        )


class MaxSpeedProposer(ModelProposer):
    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        model_speeds = [
            (model_name, MODEL_HEURISTICS.get(model_name, MODEL_HEURISTICS["gpt-4-turbo"]).tokens_per_second)
            for model_name in models_to_select
        ]

        if not model_speeds:
            return state

        max_speed = max(speed for _, speed in model_speeds)

        return state.emplaced(
            selected_models={
                model_name: {SelectionCriteria.MAX_SPEED: speed / max_speed} for model_name, speed in model_speeds
            },
            all_models=models_to_select,
        )
