from typing import Any

from mabwiser.mab import MAB

from backend.llm.ranking import DEFAULT_COST, Ranker


class MultiArmedBanditRanker(Ranker):
    """A multi-armed bandit ranker that selects arms based on expected rewards and costs."""

    rank_name = "mab"

    def __init__(
        self,
        learning_policy: Any,
        models: list[Any] | None = None,
        costs: list[float] | None = None,
        neighborhood_policy: Any = None,
        seed: int = 123,
        n_jobs: int = 1,
    ):
        """Initialize the ranker.

        Args:
            learning_policy: A learning policy for the MAB.
            models: A list of models to rank.
            costs: A list of costs for the models.
            neighborhood_policy: A neighborhood policy for the MAB.
            seed: A seed for the MAB.
            n_jobs: The number of jobs to run in parallel.
        """
        super().__init__(models, costs)
        self.mab = MAB(
            arms=list(self.models),
            learning_policy=learning_policy,
            neighborhood_policy=neighborhood_policy,
            seed=seed,
            n_jobs=n_jobs,
            backend=None,
        )

    def fit(self, model_pairs: list[tuple[str, str]], results: list[float]) -> None:
        """Train the router based on historical data."""
        models, rewards = self._pairs_to_model_rewards(model_pairs, results)
        self.mab.fit(models, rewards)

    def _pairs_to_model_rewards(
        self, model_pairs: list[tuple[str, str]], results: list[float]
    ) -> tuple[list[str], list[float]]:
        """Convert a list of model pairs to a list of models and their rewards."""
        models: list[str] = []
        rewards: list[float] = []
        for model_pair, result in zip(model_pairs, results, strict=True):
            models.extend(model_pair)
            rewards.extend([result, 1 - result])
        return models, rewards

    def _update(self, models: list[str], rewards: list[float]) -> None:
        """Update the router based on new data."""
        self.mab.partial_fit(models, rewards)

    def add_model(self, model: str, cost: float = DEFAULT_COST) -> None:
        """Add a model to the router."""
        super().add_model(model, cost)
        self.mab.add_arm(model)

    def update(self, model_a: str, model_b: str, result: float, category: str | None = None) -> None:
        """Update the ranker with a result of a battle."""
        return self.update_batch([(model_a, model_b)], [result])

    def update_batch(self, model_pairs: list[tuple[str, str]], results: list[float]) -> None:
        """Update the ranker with multiple results of a battle."""
        models, rewards = self._pairs_to_model_rewards(model_pairs, results)
        return self._update(models, rewards)

    def get_rating(self, model: str) -> float | None:
        """Return the relative rank of a model, compared to others."""
        return self.get_ratings().get(model)

    def get_ratings(self) -> dict[str, float]:
        """Return the ratings of the models."""
        return dict(self.mab.predict_expectations())

    def annotate_model(self, model: str) -> str:
        """Return an annotation for a model."""
        # self.mab doesn't store information useful for annotation.
        return ""

    def annotate_prediction(self, model_a: str, model_b: str) -> str:
        """Return an annotation for a prediction."""
        return ""
