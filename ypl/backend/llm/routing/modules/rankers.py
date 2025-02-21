import logging
from collections import OrderedDict, deque

from ypl.backend.llm.db_helpers import deduce_model_speed_scores, deduce_original_providers, deduce_semantic_groups
from ypl.backend.llm.routing.modules.base import RouterModule
from ypl.backend.llm.routing.policy import SelectionCriteria
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.routing.router_state import RouterState
from ypl.backend.llm.utils import is_yapp_model
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

    def __init__(self, only_first_n: int) -> None:
        super().__init__(name="speedRanker")
        self.only_first_n = only_first_n  # only rerank the first n results.

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

        first_part = model_names[: self.only_first_n]
        second_part = model_names[self.only_first_n :]
        sorted_first = sorted(
            first_part,
            key=lambda x: model_speed_scores.get(x, 0),  # sort by speed score
            reverse=True,
        )
        return sorted_first + second_part

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return self._select_models(state)


class YappReranker(Reranker):
    """
    If the yapp model doesn't make to the top num_models (primary models), pull it to the end, don't leave it in the
    fallback (except for cases where there's not enough models).
    TODO(Tian): right now we don't boost Yapps to the primary group, their score should help them, but this can change.
    """

    def __init__(self, num_models: int) -> None:
        super().__init__(name="yappRanker")
        self.num_models = num_models

    def rerank(self, model_names: list[str], state: RouterState) -> list[str]:
        # Split into first num_models and rest
        first_part = model_names[: self.num_models]
        rest_part = model_names[self.num_models :]

        # Find any yapp models in rest
        yapp_models = [m for m in rest_part if is_yapp_model(m)]  # noqa: F821
        if yapp_models:
            # Remove yapp models from rest
            rest_part = [m for m in rest_part if not is_yapp_model(m)]
            # Add yapp models to the very end
            rest_part.extend(yapp_models)
            model_names = first_part + rest_part
        return model_names


class PositionMatchReranker(Reranker):
    """
    Rank models so if a previous-turn reappears, make it match the position of the previous turn.
    If there was a preferred model from last turn, it will be moved to the front (even if it breaks other order).
    """

    def __init__(self, preference: RoutingPreference, only_first_n: int) -> None:
        super().__init__(name="posMatch")
        self.preference = preference
        self.only_first_n = only_first_n

    def rerank(self, model_names: list[str], state: RouterState) -> list[str]:
        if not self.preference.turns:
            return model_names

        # Get last turn's models and preferred model
        last_turn = self.preference.turns[-1]

        # split the models into ranking part and non-ranking part
        ranking_part = model_names[: self.only_first_n]
        non_ranking_part = model_names[self.only_first_n :]

        # Find any overlapping models between current models and previous turn
        overlapping = [m for m in ranking_part if m in last_turn.shown_models]
        if not overlapping:
            return model_names

        # Get positions of overlapping models in previous turn
        positions = {m: last_turn.shown_models.index(m) for m in overlapping}

        # Initialize reranked list with non-overlapping models
        reranked = [m for m in ranking_part if m not in overlapping]

        # Insert overlapping models at their previous positions
        # If position is beyond list length, append to end
        for model in sorted(overlapping, key=lambda m: positions[m]):
            pos = positions[model]
            if pos < len(reranked):
                reranked.insert(pos, model)
            else:
                reranked.append(model)

        if last_turn.preferred is not None and last_turn.preferred in reranked:
            # If there was a preferred model from last turn and it's in the current selection,
            # move it to the front
            reranked.remove(last_turn.preferred)
            reranked.insert(0, last_turn.preferred)

        return reranked + non_ranking_part

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


class AttributeScatterer(Reranker):
    """
    Scatter models with a certain attribute so they are not next to each other.
    """

    DEFAULT_DISTANCE = 4  # keep models with the same SG at least this distance apart if possible

    def __init__(self, name: str, min_dist: int | None) -> None:
        super().__init__(name=name)
        self.min_dist = min_dist or self.DEFAULT_DISTANCE

    def get_attr_map(self, models: list[str]) -> dict[str, str]:
        raise NotImplementedError

    def rerank(self, model_names: list[str], state: RouterState) -> list[str]:
        attr_map = self.get_attr_map(model_names)

        recent_groups: deque[str] = deque(maxlen=self.min_dist)
        reranked: list[str] = []
        deferred: list[str] = []

        # Process each model
        for model in model_names:
            group = attr_map.get(model)

            # If no attribute or the attribute group is not recently seen, add to result
            if group is None or group not in recent_groups:
                reranked.append(model)
                if group is not None:
                    recent_groups.append(group)
            # Otherwise defer to end
            else:
                deferred.append(model)

        # Add all deferred models at the end
        reranked.extend(deferred)

        return reranked


class SemanticGroupScatterer(AttributeScatterer):
    """
    Scatter models with same semantic group so they are not next to each other.
    This is needed for routings for prompts with attachments, as we don't do semantic group
    deduping for these prompts.
    """

    def __init__(self, min_dist: int | None) -> None:
        super().__init__(name="scatterSemGrp", min_dist=min_dist)

    def get_attr_map(self, models: list[str]) -> dict[str, str]:
        return deduce_semantic_groups(tuple(models))


class ProviderScatterer(AttributeScatterer):
    """
    Scatter models with same semantic group so they are not next to each other.
    """

    def __init__(self, min_dist: int | None) -> None:
        super().__init__(name="scatterProvider", min_dist=min_dist)

    def get_attr_map(self, models: list[str]) -> dict[str, str]:
        return deduce_original_providers(tuple(models))
