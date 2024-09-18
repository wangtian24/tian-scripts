import heapq
from abc import ABC, abstractmethod
from functools import cache
from typing import Any

import numba
import numpy as np

from ypl.backend.llm.constants import FRONTEND_MODELS
from ypl.backend.llm.ranking import Ranker, get_ranker
from ypl.backend.llm.routing.policy import DEFAULT_ROUTING_POLICY, RoutingPolicy, SelectionCriteria


class Router(ABC):
    def __init__(self, models: list[Any], policy: RoutingPolicy):
        self.models = set(models)
        self.policy = policy

    @abstractmethod
    def select_models(self, num_models: int, budget: float = float("inf"), **kwargs: Any) -> list[Any]:
        pass


class RandomRouter(Router):
    def select_models(self, num_models: int, budget: float = float("inf"), **kwargs: Any) -> list[Any]:
        return list(np.random.choice(list(self.models), size=num_models, replace=False))


class RankedRouter(Router):
    def __init__(self, models: list[Any], policy: RoutingPolicy, ranker: Ranker, seed: int = 123) -> None:
        super().__init__(models, policy)
        self.ranker = ranker
        self.rng = np.random.RandomState(seed)

    def update_ranker(self, model_a: str, model_b: str, result: float, category: str | None = None) -> None:
        self.ranker.update(model_a, model_b, result, category)
        self.models = set(self.ranker.get_models())

    def _select_models_by_minimum_traffic_fraction(self, num_models: int) -> list[Any]:
        if not num_models or not self.policy.minimum_model_traffic_fraction:
            return []

        selected_models = []
        remaining_slots = num_models

        for model, fraction in self.policy.minimum_model_traffic_fraction.items():
            if model in self.models and self.rng.random() < fraction:
                selected_models.append(model)
                remaining_slots -= 1
                if remaining_slots == 0:
                    break
        self.rng.shuffle(selected_models)
        return selected_models

    def select_models(
        self,
        num_models: int,
        budget: float = float("inf"),
        **kwargs: Any,
    ) -> list[Any]:
        """Select `num_models` models, within budget, to route to.

        Args:
            num_models: The number of models to select.
            budget: The max total budget for the models.

        Returns:
            The selected models.
        """
        if num_models > len(self.models):
            raise ValueError(f"Can't select ({num_models}) models out of {len(self.models)} available ones")

        selected_models = self._select_models_by_minimum_traffic_fraction(num_models)
        num_models_to_select = num_models - len(selected_models)

        if self.policy.random_fraction:
            for _ in range(num_models_to_select):
                if self.rng.random() < self.policy.random_fraction(self.ranker):
                    selected_models.extend(self._select_random_models(1, exclude=selected_models))

        num_models_to_select = num_models - len(selected_models)

        additional_models = []
        if self.policy.selection_criteria == SelectionCriteria.TOP:
            additional_models = self._select_best_models(num_models_to_select, budget, exclude=selected_models)
        elif self.policy.selection_criteria == SelectionCriteria.PROPORTIONAL:
            additional_models = self._select_probability_weighted_models(num_models_to_select, exclude=selected_models)
        elif self.policy.selection_criteria == SelectionCriteria.CONF_INTERVAL_WIDTH:
            additional_models = self._select_high_conf_interval_models(num_models_to_select, exclude=selected_models)
        elif self.policy.selection_criteria == SelectionCriteria.RANDOM:
            additional_models = self._select_random_models(num_models_to_select, exclude=selected_models)
        elif self.policy.selection_criteria == SelectionCriteria.CONF_INTERVAL_NUM_OVERLAP:
            additional_models = self._select_high_overlap_conf_interval_models(
                num_models_to_select, exclude=selected_models
            )
        elif self.policy.selection_criteria == SelectionCriteria.CONF_INTERVAL_PAIR_OVERLAP:
            additional_models = self._select_high_overlap_conf_interval_pair_models(
                num_models_to_select, exclude=selected_models
            )
        else:
            raise ValueError(f"Unsupported selection criteria: {self.policy.selection_criteria}")

        selected_models.extend(additional_models)
        self.rng.shuffle(selected_models)
        return selected_models

    def _select_probability_weighted_models(self, num_models: int, exclude: list[Any] | None = None) -> list[Any]:
        if not num_models:
            return []

        probabilities = self.ranker.get_probabilities().copy()
        if exclude is not None:
            for model in exclude:
                if model in probabilities:
                    del probabilities[model]
        if not probabilities:
            return []
        # Re-normalize probabilities to sum to 1
        probabilities = {k: v / sum(probabilities.values()) for k, v in probabilities.items()}

        models = list(probabilities.keys())
        return list(self.rng.choice(models, size=num_models, p=list(probabilities.values()), replace=False))

    def _select_best_models(self, num_models: int, budget: float, exclude: list[Any] | None = None) -> list[Any]:
        if not num_models:
            return []

        # Add a small random number to the rating to break ties.
        ratings = {k: v + self.rng.random() * 1e-10 for k, v in self.ranker.get_ratings().items()}
        if exclude is not None:
            for model in exclude:
                if model in ratings:
                    del ratings[model]

        sorted_models = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        if not sorted_models:
            return []

        selected_models: list[Any] = []
        total_cost = 0.0

        for model, _ in sorted_models:
            cost = self.ranker.costs[model]
            if total_cost + cost <= budget and len(selected_models) < num_models:
                selected_models.append(model)
                total_cost += cost

        if len(selected_models) < num_models:
            raise ValueError(f"Budget too low to select {num_models} models")

        return selected_models

    def _select_high_conf_interval_models(self, num_models: int, exclude: list[Any] | None = None) -> list[Any]:
        if not hasattr(self.ranker, "get_confidence_intervals"):
            raise ValueError("Ranker must implement `get_confidence_intervals`")
        conf_interval_widths = {model: abs(x[1] - x[0]) for model, x in self.ranker.get_confidence_intervals().items()}
        if exclude is not None:
            for model in exclude:
                if model in conf_interval_widths:
                    del conf_interval_widths[model]

        sorted_models = sorted(conf_interval_widths.items(), key=lambda x: x[1], reverse=True)
        return [model for model, _ in sorted_models[:num_models]]

    def _select_high_overlap_conf_interval_pair_models(
        self, num_models: int, exclude: list[Any] | None = None
    ) -> list[Any]:
        if not hasattr(self.ranker, "get_confidence_intervals"):
            raise ValueError("Ranker must implement `get_confidence_intervals`")

        conf_intervals = []
        models = []
        exclude = exclude or []
        exclude_set = set(exclude)

        for model, ci in self.ranker.get_confidence_intervals().items():
            if model in exclude_set:
                continue

            conf_intervals.append(ci)
            models.append(model)

        if not conf_intervals or not num_models:
            return []

        highest_overlapping_pairs, _ = _fast_compute_all_conf_overlap_diffs(np.array(conf_intervals), num_models)
        sorted_ind = list(dict.fromkeys(highest_overlapping_pairs.flatten()))[:num_models]

        return [models[i] for i in sorted_ind]

    def _select_random_models(self, num_models: int, exclude: list[Any] | None = None) -> list[Any]:
        models = set(self.models)
        if exclude is not None:
            models -= set(exclude)
        return list(self.rng.choice(list(models), size=num_models, replace=False))

    def _select_high_overlap_conf_interval_models(self, num_models: int, exclude: list[Any] | None = None) -> list[Any]:
        if not hasattr(self.ranker, "get_confidence_intervals"):
            raise ValueError("Ranker must implement `get_confidence_intervals`")

        conf_intervals = []
        models = []
        exclude = exclude or []
        exclude_set = set(exclude)

        for model, ci in self.ranker.get_confidence_intervals().items():
            if model in exclude_set:
                continue

            conf_intervals.append(ci)
            models.append(model)

        if not conf_intervals:
            return []

        num_overlaps, perm_map = _fast_compute_all_num_intersections(np.array(conf_intervals))
        sorted_ind = np.argpartition(num_overlaps, -num_models)[-num_models:]
        sorted_models = [(num_overlaps[i], models[perm_map[i]]) for i in sorted_ind]
        sorted_models = sorted(sorted_models, key=lambda x: x[0], reverse=True)

        return [model for _, model in sorted_models][:num_models]


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


@cache
def get_router() -> RankedRouter:
    return RankedRouter(models=FRONTEND_MODELS, policy=DEFAULT_ROUTING_POLICY, ranker=get_ranker())
