from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from backend.llm.ranking import Ranker
from backend.llm.routing.policy import RoutingPolicy, SelectionCriteria


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

    def update_ranker(self, model_a: str, model_b: str, result: float) -> None:
        self.ranker.update(model_a, model_b, result)
        self.models = set(self.ranker.models)

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
                if self.rng.random() < self.policy.random_fraction:
                    selected_models.extend(self._select_random_models(1, exclude=selected_models))

        num_models_to_select = num_models - len(selected_models)

        additional_models = []
        if self.policy.selection_criteria == SelectionCriteria.TOP:
            additional_models = self._select_best_models(num_models_to_select, budget, exclude=selected_models)
        elif self.policy.selection_criteria == SelectionCriteria.PROPORTIONAL:
            additional_models = self._select_probability_weighted_models(num_models_to_select, exclude=selected_models)
        elif self.policy.selection_criteria == SelectionCriteria.CONF_INTERVAL:
            additional_models = self._select_high_conf_interval_models(num_models_to_select, exclude=selected_models)
        elif self.policy.selection_criteria == SelectionCriteria.RANDOM:
            additional_models = self._select_random_models(num_models_to_select, exclude=selected_models)
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

    def _select_random_models(self, num_models: int, exclude: list[Any] | None = None) -> list[Any]:
        models = set(self.models)
        if exclude is not None:
            models -= set(exclude)
        return list(self.rng.choice(list(models), size=num_models, replace=False))
