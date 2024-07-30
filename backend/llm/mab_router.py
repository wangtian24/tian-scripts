from typing import Any

from mabwiser.mab import MAB


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
            self.costs = [1.0] * len(arms)
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

    def select_arms(self, num_arms: int, budget: float = float('inf')) -> list[Any]:
        """Select `num_arms` arms, within budget, to route to."""
        if num_arms > len(self.arms):
            raise ValueError(f"Can't select ({num_arms}) arms out of {len(self.arms)} available ones")

        selected_arms: list[Any] = []
        total_cost = 0.0

        expectations = self.mab.predict_expectations()

        sorted_arms = sorted(expectations.items(), key=lambda x: x[1], reverse=True)

        for arm, _ in sorted_arms:
            cost = self.costs[self.arms.index(arm)]
            if total_cost + cost <= budget and len(selected_arms) < num_arms:
                selected_arms.append(arm)
                total_cost += cost

        if len(selected_arms) < num_arms:
            raise ValueError(f"Budget too low to select {num_arms} arms")

        return selected_arms
