from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np

from backend.llm.ranking import ConfidenceIntervalRankerMixin, Ranker
from backend.llm.utils import norm_softmax


class RoutingPolicy(Enum):
    """Policy for selecting models to route to."""

    # Select the top models based on the expected reward.
    TOP = "top"

    # Select models with a probability proportional to their expected reward.
    PROPORTIONAL = "proportional"

    # Select models with larger confidence intervals.
    DECREASE_CONF_INTERVAL = "decrease_conf_interval"


class Router(ABC):
    def __init__(self, models: list[Any]):
        self.models = models

    @abstractmethod
    def select_models(
        self, num_models: int, policy: RoutingPolicy, budget: float = float("inf"), **kwargs: Any
    ) -> list[Any]:
        pass


class RandomRouter(Router):
    def select_models(
        self, num_models: int, policy: RoutingPolicy, budget: float = float("inf"), **kwargs: Any
    ) -> list[Any]:
        return list(np.random.choice(self.models, size=num_models, replace=False))


class RankedRouter(Router):
    def __init__(self, models: list[Any], ranker: Ranker, seed: int = 123) -> None:
        super().__init__(models)
        self.ranker = ranker
        self.rng = np.random.RandomState(seed)

    def update_ranker(self, model_a: str, model_b: str, result: float) -> None:
        return self.ranker.update(model_a, model_b, result)

    def select_models(
        self, num_models: int, policy: RoutingPolicy = RoutingPolicy.TOP, budget: float = float("inf"), **kwargs: Any
    ) -> list[Any]:
        """Select `num_models` models, within budget, to route to.

        Args:
            num_models: The number of models to select.
            budget: The max total budget for the models.
            policy: The policy to use for selecting models: "best" for selecting the best models within budget,
                "probability" for selecting models based on probability weighted by expected reward.

        Returns:
            The selected models.
        """
        if num_models > len(self.models):
            raise ValueError(f"Can't select ({num_models}) models out of {len(self.models)} available ones")

        if policy == RoutingPolicy.TOP:
            return self._select_best_models(num_models, budget)
        elif policy == RoutingPolicy.PROPORTIONAL:
            return self._select_probability_weighted_models(num_models, budget)
        elif policy == RoutingPolicy.DECREASE_CONF_INTERVAL:
            return self._select_decrease_conf_interval_models(num_models, budget)
        else:
            raise ValueError(f"Unsupported routing policy: {policy}")

    def _select_probability_weighted_models(self, num_models: int, budget: float) -> list[Any]:
        ranks = self.ranker.ranks()

        # Convert ranks to probabilities with a sigmoid.
        probabilities = norm_softmax(np.array(list(ranks.values())))
        models = list(ranks.keys())
        return list(self.rng.choice(models, size=num_models, p=probabilities, replace=False))

    def _select_best_models(self, num_models: int, budget: float) -> list[Any]:
        ranks = self.ranker.ranks()
        sorted_models = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

        selected_models: list[Any] = []
        total_cost = 0.0
        costs = self.ranker.costs

        for model, _ in sorted_models:
            cost = costs[self.models.index(model)]
            if total_cost + cost <= budget and len(selected_models) < num_models:
                selected_models.append(model)
                total_cost += cost

        if len(selected_models) < num_models:
            raise ValueError(f"Budget too low to select {num_models} models")

        return selected_models

    def _select_decrease_conf_interval_models(self, num_models: int, budget: float) -> list[Any]:
        if not isinstance(self.ranker, ConfidenceIntervalRankerMixin):
            raise ValueError("Ranker must be a confidence interval ranker")
        conf_intervals = self.ranker.confidence_intervals()
        sorted_models = sorted(conf_intervals.items(), key=lambda x: abs(x[1][1] - x[1][0]), reverse=True)
        return [model for model, _ in sorted_models[:num_models]]
