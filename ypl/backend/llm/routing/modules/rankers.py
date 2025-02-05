import logging
from collections import OrderedDict

from ypl.backend.llm.db_helpers import deduce_model_speed_scores
from ypl.backend.llm.routing.modules.base import RouterModule
from ypl.backend.llm.routing.policy import SelectionCriteria
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.routing.router_state import RouterState
from ypl.backend.utils.json import json_dumps
from ypl.utils import RNGMixin


class Reranker(RouterModule):
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
        return state.emplaced(selected_models=OrderedDict(selected_models))  # store using an ordered dict

    def rerank(self, model_names: list[str], state: RouterState) -> list[str]:
        """
        Rerank models. The state is also passed in to allow using some previous data.
        """
        raise NotImplementedError

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return self._select_models(state)


class ScoreReranker(Reranker):
    """
    Rank models based on a score computed from SelectionCriteria and potentially other factors.
    """

    def __init__(self) -> None:
        super().__init__(name="scoreRanker")

    def rerank(self, model_names: list[str], state: RouterState) -> list[str]:
        # Compute the score for all models.
        # NOTE(Tian) This is intentionally kept as a loop so we can add
        # other scoring factors in the futurs.

        model_scores = []
        for model in model_names:
            selection_criteria_score = sum(state.selected_models[model].values())

            score = selection_criteria_score
            model_scores.append((model, score))

        # update debug info, store the score in journey
        for model, score in model_scores:
            if model not in state.model_journey:
                state.model_journey[model] = ""
            state.model_journey[model] += f" score={score:.3f}"

        # sort by scores and return
        return [model for model, score in sorted(model_scores, key=lambda x: x[1], reverse=True)]


class SpeedReranker(Reranker):
    """
    Rank faster speed models to the front to the list.
    """

    def __init__(self) -> None:
        super().__init__(name="speedRanker")

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


class PositionMatchReranker(Reranker):
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


class PromotionModelReranker(Reranker, RNGMixin):
    """
    Ranks models based on the promotion probability.
    """

    def __init__(self) -> None:
        super().__init__(name="promoRanker")

    def rerank(self, model_names: list[str], state: RouterState) -> list[str]:
        promoted_models = []
        injected_models = []
        regular_models = []

        for model in model_names:
            if model in state.selected_models:
                criteria_map = state.selected_models[model]
                if SelectionCriteria.INJECT in criteria_map:
                    injected_models.append(model)
                elif SelectionCriteria.PROMOTED_MODELS in criteria_map:
                    promoted_models.append(model)
                else:
                    regular_models.append(model)
            else:
                regular_models.append(model)
        # If no promoted models, return original order
        if not promoted_models:
            return model_names

        # Reorder with injected first, then promoted, then rest
        return injected_models + promoted_models + regular_models

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return self._select_models(state)
