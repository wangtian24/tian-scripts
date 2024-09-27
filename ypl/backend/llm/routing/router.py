import heapq
import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from functools import cache
from typing import Any, Literal

import numba
import numpy as np
from google.cloud import logging as goog_logging

from ypl.backend.config import settings
from ypl.backend.llm.constants import COSTS_BY_MODEL, FRONTEND_MODELS
from ypl.backend.llm.ranking import Ranker, get_ranker
from ypl.backend.llm.routing.policy import DEFAULT_ROUTING_POLICY, RoutingPolicy, SelectionCriteria

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        self._running_costs: Counter = Counter()
        self._running_cost_alpha = 0.95  # EMA alpha
        self._total_costs: Counter = Counter()

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
        selected_models, selection_criterias = self._select_models_helper(num_models, budget, **kwargs)

        if settings.ROUTING_DO_LOGGING:
            decision = RoutingDecision(
                candidate_model_names=list(self.models),
                chosen_model_names=selected_models,
                selection_criteria=selection_criterias,
            )

            decision.log(structured_output=True)

        return selected_models

    def _select_models_helper(
        self,
        num_models: int,
        budget: float = float("inf"),
        always_route_good_models: bool | None = None,
        exclude: list[Any] | None = None,
        do_min_traffic_fraction: bool = True,
        **kwargs: Any,
    ) -> tuple[list[Any], list[str]]:
        """Select `num_models` models, within budget, to route to. This is a helper function for
        :py:meth:`.select_models` to account for the recursive call to include at least one good model.

        Args:
            num_models: The number of models to select.
            budget: The max total budget for the models.
            always_route_good_models: Whether to always include one good model in the selected models. See
                :py:attr:`.ROUTING_GOOD_MODELS_ALWAYS`.
            exclude: A list of models to exclude from the selection.
            do_min_traffic_fraction: Whether to include models based on the minimum traffic fraction. See
                :py:attr:`.ROUTING_MIN_TRAFFIC_FRACTION`.
            **kwargs: Additional arguments to pass to the helper function.

        Returns:
            The selected models and the selection criteria.
        """
        exclude = list(dict.fromkeys(exclude or []))

        if always_route_good_models is None:
            always_route_good_models = settings.ROUTING_GOOD_MODELS_ALWAYS

        if num_models > len(self.models):
            raise ValueError(f"Can't select ({num_models}) models out of {len(self.models)} available ones")

        selected_models = []
        selection_criterias = []

        if do_min_traffic_fraction:
            selected_models += self._select_models_by_minimum_traffic_fraction(num_models)
            selection_criterias += [SelectionCriteria._MIN_TRAFFIC_FRACTION.value] * len(selected_models)

        num_models_to_select = max(num_models - len(selected_models), 0)

        if always_route_good_models:
            num_good_to_select = min(num_models_to_select, settings.ROUTING_GOOD_MODELS_RANK_THRESHOLD)

            if num_good_to_select:
                good_models = set(
                    self._select_best_models(num_good_to_select, budget, exclude=selected_models + exclude)
                )
                bad_models = list(self.models - good_models)
                exclude_models = list(dict.fromkeys(selected_models + bad_models))

                selected_good_models, _ = self._select_models_helper(
                    num_good_to_select,
                    budget,
                    always_route_good_models=False,
                    exclude=exclude_models,
                    do_min_traffic_fraction=False,
                )

                selected_models.append(self.rng.choice(selected_good_models))
                selection_criterias.append(SelectionCriteria._ALWAYS_INCLUDE_TOP.value)

        if self.policy.random_fraction:
            for _ in range(num_models_to_select):
                if self.rng.random() < self.policy.random_fraction(self.ranker):
                    selected_models.extend(self._select_random_models(1, exclude=selected_models + exclude))
                    selection_criterias.append(SelectionCriteria.RANDOM.value)

        num_models_to_select = max(num_models - len(selected_models), 0)
        additional_models = []

        match self.policy.selection_criteria:
            case SelectionCriteria.TOP:
                additional_models = self._select_best_models(
                    num_models_to_select, budget, exclude=selected_models + exclude
                )
            case SelectionCriteria.PROPORTIONAL:
                additional_models = self._select_probability_weighted_models(
                    num_models_to_select, exclude=selected_models + exclude
                )
            case SelectionCriteria.CONF_INTERVAL_WIDTH:
                additional_models = self._select_high_conf_interval_models(
                    num_models_to_select, exclude=selected_models + exclude
                )
            case SelectionCriteria.RANDOM:
                additional_models = self._select_random_models(num_models_to_select, exclude=selected_models + exclude)
            case SelectionCriteria.MIN_RUNNING_COST:
                additional_models = self._select_running_cost_models(
                    num_models_to_select, mode="min", exclude=selected_models + exclude
                )
            case SelectionCriteria.MAX_RUNNING_COST:
                additional_models = self._select_running_cost_models(
                    num_models_to_select, mode="max", exclude=selected_models + exclude
                )
            case SelectionCriteria.CONF_INTERVAL_NUM_OVERLAP:
                additional_models = self._select_high_overlap_conf_interval_models(
                    num_models_to_select, exclude=selected_models + exclude
                )
            case SelectionCriteria.CONF_INTERVAL_PAIR_OVERLAP:
                additional_models = self._select_high_overlap_conf_interval_pair_models(
                    num_models_to_select, exclude=selected_models + exclude
                )
            case SelectionCriteria.MIN_SIMPLE_COST:
                additional_models = self._select_simple_cost_models(
                    num_models_to_select, mode="min", exclude=selected_models + exclude
                )
            case SelectionCriteria.MAX_SIMPLE_COST:
                additional_models = self._select_simple_cost_models(
                    num_models_to_select, mode="max", exclude=selected_models + exclude
                )
            case _:
                raise ValueError(f"Unsupported selection criteria: {self.policy.selection_criteria}")

        selection_criterias.extend([self.policy.selection_criteria.value] * len(additional_models))
        selected_models.extend(additional_models)

        models_to_criterias = list(zip(selected_models, selection_criterias, strict=False))
        self.rng.shuffle(models_to_criterias)

        if not models_to_criterias:
            return [], []

        selected_models_, selection_criterias_ = list(zip(*models_to_criterias, strict=False))

        return list(selected_models_), list(selection_criterias_)

    def update_running_cost(
        self,
        model: Any,
        input_str: str = "",
        output_str: str = "",
        num_input_tokens: int = 0,
        num_output_tokens: int = 0,
    ) -> None:
        """Update the running cost of a model."""

        try:
            cost_model = COSTS_BY_MODEL[model]
        except KeyError:
            # Assume the cost model is gpt4o-mini if the model is not found, which is roughly in line
            cost_model = COSTS_BY_MODEL["gpt-4o-mini"]

        c = cost_model.compute_cost(
            input_string=input_str,
            output_string=output_str,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
        )

        # we store the negative cost to make computing the min cost more efficient (Counter's most_common routine)
        c = -c

        old_val = float(self._running_costs[model] if model in self._running_costs else c)

        # mypy hates this, but it's 100% safe (float vs. int for sorting)
        self._running_costs[model] = (
            self._running_cost_alpha * old_val + (1 - self._running_cost_alpha) * c  # type: ignore
        )

        # and this as well
        self._total_costs[model] += -c  # type: ignore

    def get_total_cost(self) -> float:
        return sum(self._total_costs.values())

    def _select_running_cost_models(
        self, num_models: int, mode: Literal["min", "max"], exclude: list[Any] | None = None
    ) -> list[Any]:
        if not num_models:
            return []

        running_costs = self._running_costs.copy()
        exclude = exclude or []

        for model in exclude:
            if model in running_costs:
                del running_costs[model]

        selected_models = (
            running_costs.most_common(num_models) if mode == "min" else self._running_costs.most_common()[-num_models:]
        )

        return [model for model, _ in selected_models]

    def _select_simple_cost_models(
        self, num_models: int, mode: Literal["min", "max"], exclude: list[Any] | None = None
    ) -> list[Any]:
        if not num_models:
            return []

        exclude_set = set(exclude) if exclude else set()
        model_costs = sorted(
            (
                (cost.dollars_per_million_output_tokens + cost.dollars_per_million_input_tokens, name)
                for name, cost in COSTS_BY_MODEL.items()
                if name not in exclude_set
            ),
            key=lambda x: x[0],
        )

        if mode == "min":
            return [model for _, model in model_costs[:num_models]]
        else:
            return [model for _, model in model_costs[-num_models:]]

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
        max_cost = max(
            x.dollars_per_million_output_tokens + x.dollars_per_million_input_tokens for x in COSTS_BY_MODEL.values()
        )

        for model, _ in sorted_models:
            try:
                cost = self.ranker.costs[model]
            except KeyError:
                continue
            except AttributeError:
                cost = max_cost

            if total_cost + cost <= budget and len(selected_models) < num_models:
                selected_models.append(model)
                total_cost += cost

        if len(selected_models) < num_models:
            logging.warning(f"Budget too low to select {num_models} models; returning {len(selected_models)} models")

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


class RoutingDecision:
    def __init__(
        self,
        candidate_model_names: Sequence[str],
        chosen_model_names: Sequence[str],
        selection_criteria: Sequence[str],
        **kwargs: Any,
    ) -> None:
        """
        Args:
            candidate_model_names: The names of the models that were proposed.
            chosen_model_names: The names of the models that were finally chosen. Should be a subset of the above.
            selection_criteria: The names of the criteria that made the decision. If there are multiple, it means that
                multiple were involved in building the chosen model list, and they are inseparable.
            **kwargs: Additional metadata to log.
        """
        from ypl import __version__

        self.codebase_version = __version__
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


@cache
def get_router() -> RankedRouter:
    return RankedRouter(models=FRONTEND_MODELS, policy=DEFAULT_ROUTING_POLICY, ranker=get_ranker())
