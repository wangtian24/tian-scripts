from typing import Any, Literal

import numpy as np
from mabwiser.mab import MAB

DEFAULT_COST = 1.0


class MABRouter:
    """A multi-armed bandit router that selects arms based on expected rewards and costs."""

    def __init__(
        self,
        arms: list[Any],
        learning_policy: Any,
        neighborhood_policy: Any = None,
        costs: list[float] | None = None,
        seed: int = 123,
        n_jobs: int = 1,
    ):
        if not costs:
            self.costs = [DEFAULT_COST] * len(arms)
        else:
            assert len(arms) == len(costs), f"Number of arms ({len(arms)}) and costs ({len(costs)}) differ"
            self.costs = costs

        self.arms = arms
        self.mab = MAB(
            arms=arms,
            learning_policy=learning_policy,
            neighborhood_policy=neighborhood_policy,
            seed=seed,
            n_jobs=n_jobs,
            backend=None,
        )

    def fit(self, arms: list[str], rewards: list[float]) -> None:
        """Train the router based on historical data."""
        self.mab.fit(arms, rewards)

    def update(self, arms: list[str], rewards: list[float]) -> None:
        """Update the router based on new data."""
        self.mab.partial_fit(arms, rewards)

    def select_arms(
        self, num_arms: int, budget: float = float("inf"), policy: Literal["best", "probability"] = "best"
    ) -> list[Any]:
        """Select `num_arms` arms, within budget, to route to.

        Args:
            num_arms: The number of arms to select.
            budget: The max total budget for the arms.
            policy: The policy to use for selecting arms: "best" for selecting the best arms within budget,
                "probability" for selecting arms based on probability weighted by expected reward.

        Returns:
            The selected arms.
        """
        if num_arms > len(self.arms):
            raise ValueError(f"Can't select ({num_arms}) arms out of {len(self.arms)} available ones")

        expectations = self.mab.predict_expectations()

        if policy == "best":
            return self._select_best_arms(expectations, num_arms, budget)
        elif policy == "probability":
            return self._select_probability_weighted_arms(expectations, num_arms, budget)

    def _select_probability_weighted_arms(
        self, expectations: dict[Any, float], num_arms: int, budget: float
    ) -> list[Any]:
        probabilities = {arm: expectations[arm] / sum(expectations.values()) for arm in expectations}
        arms = list(probabilities.keys())
        return list(np.random.choice(arms, size=2, p=list(probabilities.values()), replace=False))

    def _select_best_arms(self, expectations: dict[Any, float], num_arms: int, budget: float) -> list[Any]:
        sorted_arms = sorted(expectations.items(), key=lambda x: x[1], reverse=True)

        selected_arms: list[Any] = []
        total_cost = 0.0

        for arm, _ in sorted_arms:
            cost = self.costs[self.arms.index(arm)]
            if total_cost + cost <= budget and len(selected_arms) < num_arms:
                selected_arms.append(arm)
                total_cost += cost

        if len(selected_arms) < num_arms:
            raise ValueError(f"Budget too low to select {num_arms} arms")

        return selected_arms

    def add_arm(self, arm: str, cost: float = DEFAULT_COST) -> None:
        """Add an arm to the router."""
        self.arms.append(arm)
        self.costs.append(cost)
        self.mab.add_arm(arm)
