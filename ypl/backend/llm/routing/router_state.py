from collections import defaultdict
from itertools import chain
from typing import Any

import cachetools.func
from pydantic import BaseModel
from sqlalchemy import text

from ypl.backend.db import get_engine
from ypl.backend.llm.routing.policy import SelectionCriteria


class RouterState(BaseModel):
    """
    The current state of the routing pipeline, used to track the models that have been selected and those that
    are excluded.
    """

    selected_models: dict[str, dict[SelectionCriteria, float]] = {}  # models to selection criterias and scores
    excluded_models: set[str] = set()  # models to exclude
    all_models: set[str] = set()  # all models to choose from
    always_include: bool = False  # override exclusion while combining states
    always_include_models: set[str] = set()
    applicable_modifiers: list[str] = []  # Currently, modifiers are applicable to all selected models.

    has_more_models: bool = True

    # debug information for various models
    model_scores: dict[str, float] = defaultdict(float)
    model_journey: dict[str, str] = defaultdict(str)

    def emplaced(self, **kwargs: Any) -> "RouterState":
        """
        Creates a new RouterState with the same values as the current state, but with the specified kwargs updated.
        """
        return self.model_copy(update=kwargs, deep=True)

    def __add__(self, other: "RouterState") -> "RouterState":
        """
        Merge two RouterStates together, adding the scores of duplicate model-criteria pairs. States with the
        `always_include` flag set will also include all of their selected models, regardless of whether they
        are in the excluded sets of either states. The always included models are removed from the excluded
        sets of the final state.
        """
        merged_selected_models: dict[str, dict[SelectionCriteria, float]] = {}
        excluded_models = self.excluded_models.union(other.excluded_models)
        always_included_models = self.always_include_models.union(other.always_include_models)

        if self.always_include:
            always_included_models.update(self.selected_models.keys())

        if other.always_include:
            always_included_models.update(other.selected_models.keys())

        excluded_models = excluded_models - always_included_models

        for model, criteria_map in chain(other.selected_models.items(), self.selected_models.items()):
            if model in excluded_models:
                continue

            try:
                excluded_models.remove(model)
            except KeyError:
                pass

            for criteria, score in criteria_map.items():
                if model not in merged_selected_models:
                    merged_selected_models[model] = {}
                    merged_selected_models[model][criteria] = score
                elif criteria not in merged_selected_models[model]:
                    merged_selected_models[model][criteria] = score
                else:
                    merged_selected_models[model][criteria] += score

        # Merge model journey debug maps
        merged_debug_map = defaultdict(str)
        # Handle case where both maps have the same model
        for model, debug in self.model_journey.items():
            if model in other.model_journey:
                merged_debug_map[model] = f"{debug} + ({other.model_journey[model]})"
            else:
                merged_debug_map[model] = f"({debug})"
        # Handle models only in other's debug map
        for model, debug in other.model_journey.items():
            if model not in self.model_journey:
                merged_debug_map[model] = f"({debug})"

        return RouterState(
            selected_models=merged_selected_models,
            excluded_models=excluded_models,
            all_models=self.all_models.union(other.all_models),
            always_include_models=always_included_models,
            model_journey=merged_debug_map,
        )

    def __sub__(self, other: "RouterState") -> "RouterState":
        """Return a copy of the current state excluding the selected models in the other state."""
        state = self.deepcopy()
        state.selected_models = {
            model: criteria_map
            for model, criteria_map in state.selected_models.items()
            if model not in other.selected_models
        }
        state.all_models = state.all_models.union(other.all_models)

        # Update model journey debug by appending other's values with minus sign
        for model in state.model_journey:
            if model in other.model_journey:
                state.model_journey[model] = f"{state.model_journey[model]} -({other.model_journey[model]})"

        return state

    def deepcopy(self) -> "RouterState":
        """Return a deep copy of the RouterState."""
        return self.model_copy(deep=True)

    def get_selected_models(self) -> list[str]:
        """
        Return the models that have been selected.
        """
        return list(self.selected_models.keys())

    def get_sorted_selected_models(self, priority_models: list[str] | None = None) -> list[str]:
        """
        Return the models that have been selected, sorted by the sum of their scores, highest to lowest.
        If priority_models is provided, they are pulled to the front of the list.
        """
        sorted_models = sorted(
            self.get_selected_models(), key=lambda x: sum(self.selected_models[x].values()), reverse=True
        )
        if priority_models:
            return [model for model in priority_models if model in sorted_models] + [
                model for model in sorted_models if model not in priority_models
            ]
        else:
            return sorted_models

    def get_selectable_models(self) -> set[str]:
        """
        Return the models that are not excluded.
        """
        return self.all_models - self.excluded_models

    def multiply_scores(self, factor: float) -> None:
        """
        Multiply the scores of all selected models by a factor.
        """
        for model, criteria_map in self.selected_models.items():
            for criteria, score in criteria_map.items():
                self.selected_models[model][criteria] = score * factor

    def offset_scores(self, offset: float) -> None:
        """
        Offset the scores of all selected models by a constant amount.
        """
        for model, criteria_map in self.selected_models.items():
            for criteria, score in criteria_map.items():
                self.selected_models[model][criteria] = score + offset

    def get_model_score_map(self) -> dict[str, float]:
        """
        Return a map of the models to their scores, summed over all criteria.
        """
        return {model: sum(criteria_map.values()) for model, criteria_map in self.selected_models.items()}

    def update_scores(self) -> None:
        """
        Update model_scores by summing the criteria map values for each model.
        Note that this only updates model_score for models that have been selected,
        all other scores will just be kept as is for their historic information.
        """
        for model, criteria_map in self.selected_models.items():
            self.model_scores[model] = sum(criteria_map.values())

    @classmethod
    def new_all_models_state(cls) -> "RouterState":
        rs = RouterState(
            selected_models={},
            excluded_models=set(),
            all_models=cls.get_all_models(),
        )
        rs.model_journey = {model: "" for model in rs.all_models}
        return rs

    @classmethod
    @cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)  # Cache for 10 minutes
    def get_all_models(cls) -> set[str]:
        sql_query = text(
            """
            SELECT internal_name FROM language_models
                JOIN providers ON language_models.provider_id = providers.provider_id
            WHERE language_models.deleted_at IS NULL
                AND language_models.status = 'ACTIVE'
                AND providers.deleted_at IS NULL
                AND providers.is_active IS TRUE
            """
        )

        with get_engine().connect() as conn:
            model_rows = conn.execute(sql_query)
            return set(row[0] for row in model_rows.fetchall())
