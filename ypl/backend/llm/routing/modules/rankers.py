import logging

from ypl.backend.llm.db_helpers import deduce_model_speed_scores
from ypl.backend.llm.routing.modules.base import RouterModule
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.routing.router_state import RouterState
from ypl.backend.utils.json import json_dumps


class Ranker(RouterModule):
    """
    Base reranker class that just provides a name property.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def _select_models(self, state: RouterState) -> RouterState:
        models = list(state.selected_models.keys())
        reranked_models = self.rerank(models, state)

        # Create dict mapping model name to (before_idx, after_idx) tuple, for debugging
        model_positions = {}
        for before_idx, model in enumerate(models):
            after_idx = reranked_models.index(model)
            model_positions[model] = (before_idx, after_idx)

        for model, (before_idx, after_idx) in model_positions.items():
            if model not in state.model_journey:
                state.model_journey[model] = ""
            state.model_journey[model] += f" {self.name}({before_idx}->{after_idx})"

        selected_models = [(name, state.selected_models[name]) for name in reranked_models]
        return state.emplaced(selected_models=dict(selected_models))

    def rerank(self, model_names: list[str], state: RouterState) -> list[str]:
        """
        Rerank models. The state is also passed in to allow using some previous data.
        """
        raise NotImplementedError

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return self._select_models(state)


class SpeedRanker(Ranker):
    """
    Rank faster speed models to the front to the list.
    """

    def __init__(self) -> None:
        super().__init__(name="speedReranker")

    def rerank(self, model_names: list[str], state: RouterState) -> list[str]:
        # get all model speed scores
        model_speed_scores = deduce_model_speed_scores(tuple(model_names))

        # Log and append debug info
        for model, speed_score in model_speed_scores.items():
            if model not in state.model_journey:
                state.model_journey[model] = ""
            state.model_journey[model] += f" speed={speed_score:.3f}"
        logging.info(json_dumps({"message": f"Model speed reranker: {model_speed_scores}"}))

        # sort models by speed score
        return sorted(
            model_names,
            key=lambda x: model_speed_scores.get(x, 0),  # sort by speed score
            reverse=True,
        )

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return self._select_models(state)


class PositionMatchRanker(Ranker):
    """
    Rank models so if a previous-turn reappears, make it match the position of the previous turn.
    Note that this only re-ranks the first 2 models in the input!
    """

    def __init__(self, preference: RoutingPreference) -> None:
        super().__init__(name="posMatch")
        self.preference = preference

    def rerank(self, model_names: list[str], state: RouterState) -> list[str]:
        if not self.preference.turns:
            return model_names

        # Get last turn's models and preferred model
        last_turn_models = self.preference.turns[-1].models

        # Find any overlapping models between current models and previous turn
        overlapping = [m for m in model_names if m in last_turn_models]
        if not overlapping:
            return model_names

        # Get positions of overlapping models in previous turn
        positions = {m: last_turn_models.index(m) for m in overlapping}

        # Initialize reranked list with non-overlapping models
        reranked = [m for m in model_names if m not in overlapping]

        # Insert overlapping models at their previous positions
        # If position is beyond list length, append to end
        for model in sorted(overlapping, key=lambda m: positions[m]):
            pos = positions[model]
            if pos < len(reranked):
                reranked.insert(pos, model)
            else:
                reranked.append(model)
        return reranked

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return self._select_models(state)
