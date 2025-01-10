import asyncio
import heapq
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from itertools import chain
from typing import Any

import cachetools.func
import numba
import numpy as np
from google.auth import default
from google.cloud import run_v2
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel
from sqlalchemy import func, text
from sqlmodel import Session, select

from ypl.backend.config import settings
from ypl.backend.db import get_engine
from ypl.backend.llm.chat import (
    adeduce_original_provider,
    deduce_original_providers,
    deduce_semantic_groups,
    get_all_pro_models,
    get_all_strong_models,
)
from ypl.backend.llm.constants import MODEL_HEURISTICS, ChatProvider
from ypl.backend.llm.judge import PromptModifierLabeler, YuppMultilabelClassifier, YuppOnlinePromptLabeler
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.ranking import ConfidenceIntervalRankerMixin, Ranker, get_ranker
from ypl.backend.llm.routing.policy import SelectionCriteria, decayed_random_fraction
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.vendor_langchain_adapter import GeminiLangChainAdapter, OpenAILangChainAdapter
from ypl.backend.utils.json import json_dumps
from ypl.db.language_models import LanguageModel, LanguageModelResponseStatus, LanguageModelResponseStatusEnum
from ypl.utils import RNGMixin


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

        return RouterState(
            selected_models=merged_selected_models,
            excluded_models=excluded_models,
            all_models=self.all_models.union(other.all_models),
            always_include_models=always_included_models,
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

        return state

    def deepcopy(self) -> "RouterState":
        """Return a deep copy of the RouterState."""
        return self.model_copy(deep=True)

    def get_selected_models(self) -> set[str]:
        """
        Return the models that have been selected.
        """
        return set(self.selected_models.keys())

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

    @classmethod
    @cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)  # Cache for 10 minutes
    def new_all_models_state(cls) -> "RouterState":
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
            models = set(row[0] for row in model_rows.fetchall())

        return RouterState(
            selected_models={},
            excluded_models=set(),
            all_models=models,
        )


class RouterModule(ABC):
    """
    Abstract base class for a module in the routing pipeline. Routing is defined as passing a state through an
    arithmetic circuit of modules, which may include selection, filtering, mapping, branching, and sequential
    and parallel composition, akin to MapReduce or a Unix pipeline. A router is composed of a sequence of
    arithmetic operations on RouterModule, e.g.,

    .. code-block::python

        router = CostModelProposer() | TopK(3)
        state = router.select_models(state=RouterState(all_models=set(FRONTEND_MODELS)))
        print(state.get_selected_models())

    would route to the top 3 cheapest models out of the available front-end models, passing the state
    sequentially through CostModelProposer to TopK. These three arithmetic operations are supported:

    .. code-block::python

        # Sequential composition; the state is passed sequentially through each module
        router = CostModelProposer() | EloProposer(ranker)

        # Parallel composition; the state is split across two modules, which operate in parallel.
        # The results of each module are merged together (see :py:meth:`.RouterState.__add__`).
        router = CostModelProposer() & EloProposer(ranker)

        # Exclusive drawing; the state is randomly routed to one of the two modules, with a probability
        # given by the `with_probs` method.
        router = CostModelProposer() ^ EloProposer(ranker)

    These modules can be chained and combined arbitrarily, e.g.,

    .. code-block::python

        router = (
            (((EloProposer(ranker) & CostModelProposer()) | TopK(3))
            ^ ConfidenceIntervalWidthModelProposer(ranker)).with_probs(0.75, 0.25)
        ) | RoutingDecisionLogger()

    See Also:
    - :py:meth:`.RouterSequentialChain`, :py:meth:`.RouterParallelChain`, :py:meth:`.RouterExclusiveChain`
    - :py:meth:`.RouterState.__add__`, :py:meth:`.RouterState.multiply_scores`
    - :py:meth:`.RoutingDecisionLogger`
    """

    _multiplier: float | None = None
    _always_include: bool | None = None
    _offset: float | None = None

    def select_models(self, state: RouterState | None = None) -> RouterState:
        response = self._select_models(state or RouterState())

        if self._multiplier is not None:
            response.multiply_scores(self._multiplier)

        if self._offset is not None:
            response.offset_scores(self._offset)

        if self._always_include is not None:
            response.always_include = self._always_include

        return response

    async def aselect_models(self, state: RouterState | None = None) -> RouterState:
        response = await self._aselect_models(state or RouterState())

        if self._multiplier is not None:
            response.multiply_scores(self._multiplier)

        if self._offset is not None:
            response.offset_scores(self._offset)

        if self._always_include is not None:
            response.always_include = self._always_include

        return response

    def with_flags(
        self,
        *,
        multiplier: float | None = None,
        always_include: bool | None = None,
        offset: float | None = None,
    ) -> "RouterModule":
        if multiplier is not None:
            self._multiplier = multiplier

        if offset is not None:
            self._offset = offset

        if always_include is not None:
            self._always_include = always_include

        return self

    def __or__(self, other: "RouterModule") -> "RouterModule":
        return RouterSequentialChain(self, other)

    def __and__(self, other: "RouterModule") -> "RouterModule":
        return RouterParallelChain(self, other)

    def __xor__(self, other: "RouterModule") -> "RouterExclusiveChain":
        return RouterExclusiveChain(self, other)

    @abstractmethod
    def _select_models(self, state: RouterState) -> RouterState:
        raise NotImplementedError

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return self._select_models(state)


class Passthrough(RouterModule):
    def _select_models(self, state: RouterState) -> RouterState:
        return state

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return state


class ModifierAnnotator(RouterModule):
    """Just adds the applicable modifiers to the state."""

    def __init__(self, modifiers: list[str]) -> None:
        self.modifiers = modifiers

    def _select_models(self, state: RouterState) -> RouterState:
        return state.emplaced(applicable_modifiers=self.modifiers)

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return state.emplaced(applicable_modifiers=self.modifiers)


class ConsoleDebugPrinter(RouterModule):
    def _select_models(self, state: RouterState) -> RouterState:
        print(state)
        return state

    async def _aselect_models(self, state: RouterState) -> RouterState:
        print(state)
        return state


class RoutingDecisionLogger(RouterModule):
    def __init__(self, enabled: bool = True, prefix: str = "router", metadata: dict[str, Any] | None = None) -> None:
        """
        Args:
            enabled: Whether to log the routing decision.
        """
        self.enabled = enabled
        self.prefix = prefix
        self.metadata = metadata or {}

    def _select_models(self, state: RouterState) -> RouterState:
        """
        Log the routing decision.
        """
        if self.enabled:
            criterias = [
                (str(criteria), score)
                for _, criteria_map in state.selected_models.items()
                for criteria, score in criteria_map.items()
            ]

            decision = RoutingDecision(
                prefix=self.prefix,
                candidate_model_names=list(state.all_models),
                chosen_model_names=list(state.selected_models.keys()),
                selection_criteria=criterias,
                applicable_prompt_modifiers=state.applicable_modifiers,
                **self.metadata,
            )
            decision.log()

        return state


class RouterSequentialChain(RouterModule):
    def __init__(self, *args: RouterModule) -> None:
        """
        Represents a sequential chain of routing modules, i.e., the state is passed sequentially through each module.
        See :py:class:`.RouterModule` for more information.

        Args:
            *args: The modules to pass the state through sequentially.
        """
        self.router_modules: list[RouterModule] = list(args)

    def __or__(self, other: "RouterModule") -> "RouterModule":
        self.router_modules.append(other)
        return self

    def _select_models(self, state: RouterState) -> RouterState:
        for router_module in self.router_modules:
            state = router_module.select_models(state=state)

        return state

    async def _aselect_models(self, state: RouterState) -> RouterState:
        for router_module in self.router_modules:
            state = await router_module.aselect_models(state=state)

        return state


class RouterExclusiveChain(RNGMixin, RouterModule):
    def __init__(self, *args: RouterModule) -> None:
        """
        Represents an exclusive chain of routing modules, i.e., the state is randomly routed to one of the modules.
        See :py:class:`.RouterModule` for more information.

        Args:
            *args: The modules to randomly route the state to.
        """
        self.router_modules: list[RouterModule] = list(args)
        self.random_probabilities: list[float] = []

    def __xor__(self, other: "RouterModule") -> "RouterExclusiveChain":
        self.router_modules.append(other)
        return self

    def with_probs(self, *probabilities: float) -> "RouterExclusiveChain":
        """
        Set the probabilities of routing to each module in the same order that it was constructed.

        Args:
            *probabilities: The probabilities of routing to each module.
        """
        p = np.array(probabilities)
        p = p / p.sum()
        self.random_probabilities = p.tolist()

        return self

    def _choose_module(self) -> RouterModule:
        if len(self.random_probabilities) != len(self.router_modules):
            logging.warning(
                "Random probabilities not set for RouterExclusiveChain; using default of 1/len(router_modules)"
            )
            probs = np.full(len(self.router_modules), 1 / len(self.router_modules))
        else:
            probs = np.array(self.random_probabilities)

        return self.get_rng().choice(  # type: ignore[no-any-return]
            np.array(self.router_modules, dtype=object), replace=False, p=probs
        )

    def _select_models(self, state: RouterState) -> RouterState:
        chosen_module = self._choose_module()
        return chosen_module.select_models(state=state)

    async def _aselect_models(self, state: RouterState) -> RouterState:
        chosen_module = self._choose_module()
        return await chosen_module.aselect_models(state=state)


class RouterParallelChain(RouterModule):
    def __init__(self, *args: RouterModule) -> None:
        """
        Represents a parallel chain of routing modules, i.e., the state is split across all modules.
        See :py:class:`.RouterModule` for more information.

        Args:
            *args: The modules to pass the state through in parallel.
        """
        self.router_modules: list[RouterModule] = list(args)

    def __and__(self, other: "RouterModule") -> "RouterModule":
        self.router_modules.append(other)
        return self

    def _select_models(self, state: RouterState) -> RouterState:
        responses = []

        for router_module in self.router_modules:
            router_response = router_module.select_models(state=state.deepcopy())
            responses.append(router_response)

        for response in responses:
            state += response

        return state

    async def _aselect_models(self, state: RouterState) -> RouterState:
        responses = []

        for router_module in self.router_modules:
            router_response = await router_module.aselect_models(state=state.deepcopy())
            responses.append(router_response)

        for response in responses:
            state += response

        return state


class ModelProposer(RouterModule):
    """
    Represents a proposer of a set of models to route to. Subclasses should implement the `_propose_models` method
    to define the specific model selection logic.
    """

    def _select_models(self, state: RouterState) -> RouterState:
        """
        Propose a set of models to route to.

        Args:
            num_models: The number of models to propose.
            state: The state to propose models for. The `all_models` field is used to determine the set of
                models to propose, e.g., `FRONTEND_MODELS`.
        """
        models_to_select = state.get_selectable_models()
        response = self._propose_models(models_to_select, state.deepcopy())

        return response

    async def _aselect_models(self, state: RouterState) -> RouterState:
        models_to_select = state.get_selectable_models()
        response = await self._apropose_models(models_to_select, state.deepcopy())

        return response

    @abstractmethod
    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        """
        Propose a set of models to route to.

        Args:
            num_models: The number of models to propose.
            models_to_select: The set of models to propose.
            state: The state to propose models for.
        """
        raise NotImplementedError

    async def _apropose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        return self._propose_models(models_to_select, state)


class ProModelProposer(RNGMixin, ModelProposer):
    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        if not models_to_select:
            return RouterState()

        all_pro_models = set(get_all_pro_models())
        models_to_select = models_to_select.intersection(all_pro_models)
        models_to_select_ = list(models_to_select)
        self.get_rng().shuffle(models_to_select_)

        return state.emplaced(
            selected_models={model: {SelectionCriteria.PRO_MODELS: 1.0} for model in models_to_select_},
            all_models=state.all_models,
            excluded_models=state.excluded_models | (state.all_models - set(models_to_select)),
        )

    async def _apropose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        return self._propose_models(models_to_select, state)


class StrongModelProposer(RNGMixin, ModelProposer):
    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        if not models_to_select:
            return RouterState()

        all_strong_models = set(get_all_strong_models())
        models_to_select = models_to_select.intersection(all_strong_models)
        models_to_select_ = list(models_to_select)
        self.get_rng().shuffle(models_to_select_)

        return state.emplaced(
            selected_models={model: {SelectionCriteria.STRONG_MODELS: 1.0} for model in models_to_select_},
            all_models=state.all_models,
            excluded_models=state.excluded_models | (state.all_models - set(models_to_select)),
        )

    async def _apropose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        return self._propose_models(models_to_select, state)


class RandomModelProposer(RNGMixin, ModelProposer):
    def __init__(self, *, models: set[str] | None = None, providers: set[str] | None = None) -> None:
        self.models = models or set()
        self.providers = providers or set()

    def _random_select(self, models_to_select: set[str], state: RouterState) -> RouterState:
        if not models_to_select:
            return RouterState()

        selected_models = self.get_rng().choice(list(models_to_select), len(models_to_select), replace=False)

        return state.emplaced(
            selected_models={model: {SelectionCriteria.RANDOM: self.get_rng().random()} for model in selected_models},
            all_models=state.all_models,
            excluded_models=state.excluded_models | (state.all_models - set(selected_models)),
        )

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        """
        Randomly select a set of models to route to.

        Args:
            num_models: The number of models to propose.
            models_to_select: The set of models to randomly select from.
            state: The state to propose models for.
        """
        if self.models:
            models_to_select = models_to_select.intersection(self.models)

        if self.providers:
            provider_map = deduce_original_providers(tuple(models_to_select))
            models_to_select = {model for model in models_to_select if provider_map[model] in self.providers}

        return self._random_select(models_to_select, state)

    async def _apropose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        if self.models:
            models_to_select = models_to_select.intersection(self.models)

        if self.providers:
            models_to_select = {
                model for model in models_to_select if (await adeduce_original_provider(model)) in self.providers
            }

        return self._random_select(models_to_select, state)


class ModelFilter(RouterModule):
    """
    Represents a filter of a set of models to route to. Subclasses should implement the `_filter` method
    to define the specific model filtering logic.
    """

    def __init__(self, persist: bool = False, exempt_models: set[str] | None = None) -> None:
        """
        Args:
            persist: Whether to persist the models that are excluded from the selection, so that they are never
                selected again in future routing modules.
            exempt_models: The models to exempt from the filter.
        """
        self.persist = persist
        self.exempt_models = exempt_models or set()

    def _select_models(self, state: RouterState) -> RouterState:
        state, excluded_models = self._filter(state)
        excluded_models = excluded_models - self.exempt_models
        state.excluded_models = state.excluded_models - self.exempt_models

        if self.persist:
            state.excluded_models = state.excluded_models.union(excluded_models)

        return state

    async def _aselect_models(self, state: RouterState) -> RouterState:
        state, excluded_models = await self._afilter(state)
        excluded_models = excluded_models - self.exempt_models
        state.excluded_models = state.excluded_models - self.exempt_models

        if self.persist:
            state.excluded_models = state.excluded_models.union(excluded_models)

        return state

    @abstractmethod
    def _filter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        """
        Filter a set of models to route to. Returns the filtered state and the excluded models.

        Args:
            state: The state to filter models for.
        """
        raise NotImplementedError

    async def _afilter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        return self._filter(state)


class TopK(ModelFilter):
    """
    Represents a filter of a set of models to route to. The top-k models are selected, where the score is the sum of the
    scores of the models.
    """

    def __init__(self, k: int, persist: bool = True) -> None:
        """
        Args:
            k: The number of top-k models to select.
        """
        super().__init__(persist=persist)
        self.k = k

    def _filter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        selected_models = sorted(state.selected_models.items(), key=lambda x: sum(x[1].values()), reverse=True)
        excluded_models = {model for model, _ in selected_models[self.k :]}

        state.selected_models = {model: x for model, x in state.selected_models.items() if model not in excluded_models}

        return state, excluded_models


class RandomJitter(RNGMixin, ModelFilter):
    """Randomly jitters the sum of scores for each selected model."""

    def __init__(
        self,
        *,
        jitter_range: float | None = None,
        jitter_pct: float | None = None,
        persist: bool = True,
    ) -> None:
        """
        Creates a new RandomJitter filter.

        Args:
            jitter_range: The range of the jitter.
            jitter_pct: The percentage of the sum of scores to jitter as a number between 0 and 1.
            persist: Whether to persist the models that are excluded from the selection, so that they are never
                selected again in future routing modules.
        """
        super().__init__(persist=persist)
        self.jitter_range = jitter_range
        self.jitter_pct = jitter_pct

    def _filter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        for scores in state.selected_models.values():
            if self.jitter_range is not None:
                jitter = self.get_rng().uniform(-self.jitter_range, self.jitter_range) / len(scores)
            elif self.jitter_pct is not None:
                jitter = self.get_rng().uniform(-self.jitter_pct, self.jitter_pct)
                jitter = jitter * sum(scores.values()) / len(scores)
            else:
                raise ValueError("Either jitter_range or jitter_pct must be provided")

            for criterion, score in scores.items():
                scores[criterion] = max(0, score + jitter)

        return state, set()


class Inject(ModelFilter):
    def __init__(self, models: list[str], *, score: float = 1) -> None:
        super().__init__(persist=True)
        self.models = models
        self.score = score

    def _filter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        return state.emplaced(
            selected_models={
                **state.selected_models,
                **{model: {SelectionCriteria.INJECT: self.score - idx} for idx, model in enumerate(self.models)},
            },
        ), set()


class MinimumFractionModelProposer(RNGMixin, ModelProposer):
    def __init__(
        self,
        num_models: int,
        minimum_model_traffic_fraction: dict[str, float],
    ) -> None:
        self.minimum_model_traffic_fraction = minimum_model_traffic_fraction
        self.num_models = num_models

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        if self.num_models > len(models_to_select):
            return state

        models = list(models_to_select)
        probs = np.array([self.minimum_model_traffic_fraction.get(model, 0) for model in models])
        probs /= probs.sum()

        selected_models = self.get_rng().choice(models, min(self.num_models, len(models)), replace=False, p=probs)

        return state.emplaced(
            selected_models={model: {SelectionCriteria._MIN_TRAFFIC_FRACTION: 1.0} for model in selected_models},
            excluded_models=state.excluded_models,
        )


class RandomShuffle(RNGMixin, ModelProposer):
    def __init__(self, shuffle_same_scores_only: bool = False) -> None:
        self.shuffle_same_scores_only = shuffle_same_scores_only

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        if self.shuffle_same_scores_only:
            # Shuffle only the order of models with the same score
            model_score_map = state.get_model_score_map()
            score_model_map: dict[float, list[str]] = {score: [] for score in model_score_map.values()}
            shuffled_items = list(model_score_map.items())
            self.get_rng().shuffle(shuffled_items)  # type: ignore

            for model, score in shuffled_items:
                score_model_map[score].append(model)

            selected_models = {}

            for score, models in score_model_map.items():
                for model in models:
                    selected_models[model] = {SelectionCriteria.RANDOM: score}

            return state.emplaced(
                selected_models=selected_models, excluded_models=state.excluded_models, all_models=set(models_to_select)
            )
        else:
            selected_models = self.get_rng().choice(list(models_to_select), len(models_to_select), replace=False)  # type: ignore

            return state.emplaced(
                selected_models={model: {SelectionCriteria.RANDOM: 1.0} for model in selected_models},
                excluded_models=state.excluded_models,
                all_models=set(models_to_select),
            )


class AlwaysGoodModelMetaRouter(ModelProposer):
    """
    A meta-router that ensures that at least one "good" model is always selected, where the definition of "good" is
    defined by the `num_good` parameter and the `ranker` parameter, which is a :py:class:`.Ranker` used to rank
    the models.
    """

    def __init__(self, ranker: Ranker, router: RouterModule, num_good: int) -> None:
        self.router = router
        self.num_good = num_good
        self.ranker = ranker

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        good_model_response = (EloProposer(self.ranker) | TopK(self.num_good)).select_models(state.deepcopy())
        good_model_response.all_models = good_model_response.get_selected_models()
        good_model_response = self.router.select_models(good_model_response)

        top1_filter = TopK(1)
        good_model_response = top1_filter.select_models(state=good_model_response)
        good_model_response.multiply_scores(1000)  # ensures that one good model is always selected
        state.excluded_models.update(good_model_response.selected_models.keys())

        default_response = self.router.select_models(state.deepcopy())
        default_response.excluded_models = set()
        good_model_response.excluded_models = set()
        good_model_response += default_response

        return good_model_response


class EloProposer(RNGMixin, ModelProposer):
    """
    A proposer of a set of models to route to. The models are ranked by the `ranker` parameter, which is a
    :py:class:`.Ranker`.
    """

    def __init__(self, ranker: Ranker) -> None:
        """
        Args:
            ranker: The ranker to use to rank the models.
        """
        self.ranker = ranker

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        elo_ratings = self.ranker.get_ratings()

        selected_models = sorted(
            [model for model in models_to_select],
            key=lambda m: elo_ratings.get(m, 1.0) + self.get_rng().random() * 1e-7,  # add a small random jitter
            reverse=True,
        )

        max_elo = max(elo_ratings.values(), default=1.0)

        return state.emplaced(
            selected_models={
                model: {SelectionCriteria.TOP: elo_ratings.get(model, 1.0) / max_elo} for model in selected_models
            }
        )


class ProportionalModelProposer(RNGMixin, ModelProposer):
    """
    A proposer of a set of models to route to. The models are selected proportionally following
    :py:meth:`.Ranker.get_probabilities`.
    """

    def __init__(self, num_models: int, ranker: Ranker) -> None:
        """
        Args:
            num_models: The number of models to propose.
            ranker: The ranker to use to rank the models.
        """
        self.ranker = ranker
        self.num_models = num_models

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        probabilities = self.ranker.get_probabilities().copy()

        if not probabilities:
            return RouterState()

        for k, _ in list(probabilities.items()):
            if k not in models_to_select:
                del probabilities[k]

        if not probabilities:
            return RouterState()

        # Re-normalize probabilities to sum to 1
        probabilities = {k: v / sum(probabilities.values()) for k, v in probabilities.items()}
        models = list(probabilities.keys())

        chosen_models = set(
            self.get_rng().choice(
                models,
                size=min(self.num_models, len(models)),
                p=list(probabilities.values()),
                replace=False,
            )
        )

        return state.emplaced(
            selected_models={model: {SelectionCriteria.PROPORTIONAL: 1.0} for model in chosen_models},
            all_models=set(models),
        )


class ConfidenceIntervalWidthModelProposer(ModelProposer):
    """
    A proposer of a set of models to route to. The models are selected based on the width of the confidence
    intervals, where the width is defined by the `ranker`.
    """

    def __init__(self, num_models: int, ranker: ConfidenceIntervalRankerMixin) -> None:
        """
        Args:
            num_models: The number of models to propose.
            ranker: The ranker to use to rank the models.
        """
        self.ranker = ranker
        self.num_models = num_models

        if not hasattr(self.ranker, "get_confidence_intervals"):
            raise ValueError("Ranker must implement `get_confidence_intervals`")

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        conf_interval_widths = {model: abs(x[1] - x[0]) for model, x in self.ranker.get_confidence_intervals().items()}

        for model in list(conf_interval_widths.keys()):
            if model not in models_to_select:
                del conf_interval_widths[model]

        sorted_models = sorted(conf_interval_widths.items(), key=lambda x: x[1], reverse=True)
        selected_models = [model for model, _ in sorted_models[: self.num_models]]

        return state.emplaced(
            selected_models={
                model: {SelectionCriteria.CONF_INTERVAL_WIDTH: 1.0 / (rank + 1)}
                for rank, model in enumerate(selected_models)
            },
            all_models=set(conf_interval_widths.keys()),
        )


class ConfidenceIntervalNumOverlapModelProposer(ModelProposer):
    """
    A proposer of a set of models to route to. The models are selected based on the number of overlaps in the confidence
    intervals, where the confidence intervals are computed by the `ranker`.
    """

    def __init__(self, num_models: int, ranker: Ranker) -> None:
        """
        Args:
            ranker: The ranker to use to rank the models.
        """
        self.ranker = ranker
        self.num_models = num_models

        if not hasattr(self.ranker, "get_confidence_intervals"):
            raise ValueError("Ranker must implement `get_confidence_intervals`")

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        conf_intervals = []
        models = []

        for model, ci in self.ranker.get_confidence_intervals().items():  # type: ignore
            if model not in models_to_select:
                continue

            conf_intervals.append(ci)
            models.append(model)

        if not conf_intervals:
            return RouterState()

        num_overlaps, perm_map = _fast_compute_all_num_intersections(np.array(conf_intervals))
        k = min(self.num_models, len(models))
        sorted_ind = np.argpartition(num_overlaps, -k)[-k:]
        sorted_models = [(num_overlaps[i], models[perm_map[i]]) for i in sorted_ind]
        sorted_models = sorted(sorted_models, key=lambda x: x[0], reverse=True)

        selected_models = [model for _, model in sorted_models][:k]

        return state.emplaced(
            selected_models={
                model: {SelectionCriteria.CONF_INTERVAL_NUM_OVERLAP: 1.0 / (rank + 1)}
                for rank, model in enumerate(selected_models)
            },
            all_models=set(models),
        )


class ConfidenceIntervalWidthOverlapModelProposer(ModelProposer):
    """
    A proposer of a set of models to route to. The models are selected based on the overlap width of the
    confidence intervals, where the confidence intervals are computed by the `ranker`.
    """

    def __init__(self, num_models: int, ranker: Ranker) -> None:
        """
        Args:
            num_models: The number of models to propose.
            ranker: The ranker to use to rank the models.
        """
        self.num_models = num_models
        self.ranker = ranker

        if not hasattr(self.ranker, "get_confidence_intervals"):
            raise ValueError("Ranker must implement `get_confidence_intervals`")

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        conf_intervals = []
        models = []

        for model, ci in self.ranker.get_confidence_intervals().items():  # type: ignore
            if model not in models_to_select:
                continue

            conf_intervals.append(ci)
            models.append(model)

        num_models = min(self.num_models, len(models))

        if not conf_intervals or not num_models:
            return RouterState()

        highest_overlapping_pairs, _ = _fast_compute_all_conf_overlap_diffs(np.array(conf_intervals), num_models)
        sorted_ind = list(dict.fromkeys(highest_overlapping_pairs.flatten()))[:num_models]

        return state.emplaced(
            selected_models={
                models[i]: {SelectionCriteria.CONF_INTERVAL_PAIR_OVERLAP: 1.0 / (rank + 1)}
                for rank, i in enumerate(sorted_ind)
            },
            all_models=set(models),
        )


class CostModelProposer(ModelProposer):
    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        model_costs = []

        for model in models_to_select:
            m = model if model in MODEL_HEURISTICS else "gpt-4o-mini"
            cost = (
                MODEL_HEURISTICS[m].dollars_per_million_output_tokens
                + MODEL_HEURISTICS[m].dollars_per_million_input_tokens
            )
            model_costs.append((cost, model))

        selected_models = sorted(model_costs, key=lambda x: x[0])

        return state.emplaced(
            selected_models={
                model: {SelectionCriteria.MIN_SIMPLE_COST: 1 / (cost + 1)} for cost, model in selected_models
            },
            all_models=set([x[1] for x in model_costs]),
        )


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


class MaxSpeedProposer(ModelProposer):
    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        model_speeds = [
            (model_name, MODEL_HEURISTICS.get(model_name, MODEL_HEURISTICS["gpt-4-turbo"]).tokens_per_second)
            for model_name in models_to_select
        ]

        if not model_speeds:
            return state

        max_speed = max(speed for _, speed in model_speeds)

        return state.emplaced(
            selected_models={
                model_name: {SelectionCriteria.MAX_SPEED: speed / max_speed} for model_name, speed in model_speeds
            },
            all_models=models_to_select,
        )


class Exclude(ModelFilter):
    """
    Represents a filter to remove certain models and models from certain providers.
    """

    def __init__(self, *, models: set[str] | None = None, providers: set[str] | None = None):
        super().__init__(persist=True)
        self.models = models or set()
        self.providers = providers or set()

    def _filter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        state = state.deepcopy()
        excl_models = self.models

        if self.providers:
            provider_map = deduce_original_providers(tuple(state.selected_models.keys()))
            excl_models = excl_models | {
                model for model in state.selected_models.keys() if provider_map[model] in self.providers
            }

        state.selected_models = {k: v for k, v in state.selected_models.items() if k not in excl_models}
        state.excluded_models = state.excluded_models | excl_models

        return state, excl_models


class StreamableModelFilter(Exclude):
    """
    Represents a filter of a set of models that support streaming.
    """

    def __init__(self) -> None:
        non_streaming_models = {model for model, heuristics in MODEL_HEURISTICS.items() if not heuristics.can_stream}
        super().__init__(models=non_streaming_models)


class RoutingDecision:
    def __init__(
        self,
        prefix: str,
        candidate_model_names: Sequence[str],
        chosen_model_names: Sequence[str],
        selection_criteria: Sequence[tuple[str, float]],
        applicable_prompt_modifiers: Sequence[str],
        **kwargs: Any,
    ) -> None:
        """
        Args:
            prefix: A prefix to identify the router that made the decision.
            candidate_model_names: The names of the models that were proposed.
            chosen_model_names: The names of the models that were finally chosen. Should be a subset of the above.
            selection_criteria: The names of the criteria and scores that made the decision. If there are multiple,
                it means that multiple were involved in building the chosen model list, and they are inseparable.
            **kwargs: Additional metadata to log.
        """
        from ypl import __version__

        self.codebase_version = __version__
        self.prefix = prefix
        self.candidate_model_names = candidate_model_names
        self.chosen_model_names = chosen_model_names
        self.selection_criteria = selection_criteria
        self.applicable_prompt_modifiers = applicable_prompt_modifiers
        self.additional_metadata = kwargs

    def log(self) -> None:
        """
        Logs the routing decision with a lot of schema flexibility. Parts of this should eventually be standardized
        once our router models are more concrete. The `additional_metadata` kwargs will be stored under the
        `additional_metadata` field in the log.
        """
        log_dict = {
            "codebase_version": self.codebase_version,
            "prefix": self.prefix,
            "candidate_model_names": self.candidate_model_names,
            "chosen_model_names": self.chosen_model_names,
            "selection_criteria": self.selection_criteria,
            "applicable_prompt_modifiers": self.applicable_prompt_modifiers,
            "additional_metadata": self.additional_metadata,
        }

        logging.info(json_dumps(log_dict))


def group_models_by_key(model_to_key_map: dict[str, str], sorted_model_list: list[str]) -> dict[str | None, list[str]]:
    """
    Group models in sorted_model_list by their keys found in map, keep the order.
    All models without keys in map are grouped under 'None'.
    """
    model_list_by_key: dict[str | None, list[str]] = defaultdict(list[str])
    for model in sorted_model_list:
        key = model_to_key_map.get(model, None)
        model_list_by_key[key].append(model)
    return model_list_by_key


class ProviderFilter(ModelFilter):
    """
    Filters models based on their provider. If `one_per_provider` is set, only one model per provider is selected, e.g.
    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "claude-3-opus", "claude-3.5-sonnet"] -> ["gpt-4o", "claude-3-opus"].
    The filtering occurs in order of the providers by reverse score, so higher-scoring models are chosen first.

    If `providers` is set, only models in the set are considered. If `inverse` is set, the filter is inverted, so models
    not in the set are selected.
    """

    def __init__(
        self,
        one_per_provider: bool = False,
        providers: set[str] | None = None,
        priority_models: list[str] | None = None,
        inverse: bool = False,
        persist: bool = False,
    ) -> None:
        """
        Args:
            one_per_provider: Whether to only select one model per provider.
            providers: A set of providers to filter by. If empty, all providers are considered. If a string, it is
                split by commas and spaces.
            inverse: Whether to invert the filter.
            persist: Whether to persist the filter across requests, storing the removed models in the `excluded_models`
                field of the router state.
            priority_models: A list of models we always want to keep.
        """
        super().__init__(persist)

        if isinstance(providers, str):
            providers = {providers}

        self.providers: set[str] | None = providers
        self.inverse = inverse
        self.one_per_provider = one_per_provider
        self.priority_models = priority_models or []
        assert self.providers or self.one_per_provider, "Either providers or one_per_provider must be set"

    def should_keep_for_provider(self, provider: str | None) -> bool:
        """check if any model from this provider should be kept"""
        if provider is None:
            return False  # provider-less models are always dropped
        return not self.providers or (self.inverse ^ (provider in self.providers))

    def _filter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        sorted_models = state.get_sorted_selected_models(priority_models=self.priority_models)
        provider_map = deduce_original_providers(tuple(sorted_models))
        models_by_provider = group_models_by_key(provider_map, sorted_models)

        keep_models = set()
        for provider, models in models_by_provider.items():
            if self.should_keep_for_provider(provider):
                if not self.one_per_provider:
                    keep_models.update(models)  # every models get to stay
                else:
                    added = False
                    for model in models:
                        if self.priority_models and model in self.priority_models or not added:
                            keep_models.add(model)
                            added = True
                        else:
                            break

        excluded_models = state.selected_models.keys() - keep_models

        return state.emplaced(
            selected_models={model: state.selected_models[model] for model in keep_models}
        ), excluded_models


class OnePerSemanticGroupFilter(ModelFilter):
    """
    Filters models based on their semantic group. User selected models are always kept.
    """

    def __init__(self, persist: bool = False, priority_models: list[str] | None = None) -> None:
        super().__init__(persist=persist)
        self.priority_models = priority_models or []

    def _filter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        sorted_models = state.get_sorted_selected_models(priority_models=self.priority_models)
        semantic_group_map = deduce_semantic_groups(tuple(sorted_models))
        models_by_semantic_group = group_models_by_key(semantic_group_map, sorted_models)

        keep_models = set()
        for semantic_group, models in models_by_semantic_group.items():
            if semantic_group is None:
                keep_models.update(models)  # add all models with no semantic group
            else:
                added = False
                for model in models:
                    if (self.priority_models and model in self.priority_models) or not added:
                        keep_models.add(model)
                        added = True
                    else:
                        break

        excluded_models = state.selected_models.keys() - keep_models

        return state.emplaced(
            selected_models={model: state.selected_models[model] for model in keep_models}
        ), excluded_models


def get_router_ranker(ranker: Ranker | None = None) -> tuple[RouterModule, Ranker]:
    min_weight = settings.ROUTING_WEIGHTS.get("min_simple_cost", 0.1)
    rand_weight = settings.ROUTING_WEIGHTS.get("random", 0.1)
    top_weight = settings.ROUTING_WEIGHTS.get("top", 0.1)
    ranker = ranker or get_ranker()

    router: RouterModule = (CostModelProposer() ^ RandomModelProposer() ^ EloProposer(ranker)).with_probs(
        min_weight,
        rand_weight + decayed_random_fraction(ranker, initial_value=0.6, final_value=0.05, steps=50000),
        top_weight,
    )

    if settings.ROUTING_GOOD_MODELS_ALWAYS:
        router = AlwaysGoodModelMetaRouter(ranker, router, num_good=settings.ROUTING_GOOD_MODELS_RANK_THRESHOLD)

    router = router | TopK(2) | RoutingDecisionLogger(enabled=settings.ROUTING_DO_LOGGING, prefix="default-router")

    return router, ranker


def get_router(ranker: Ranker | None = None) -> RouterModule:
    return get_router_ranker(ranker)[0]


def get_gcp_cloud_run_uri(service_name: str, region: str) -> str:
    credentials, project_id = default()
    client = run_v2.ServicesClient(credentials=credentials)
    name = f"projects/{project_id}/locations/{region}/services/{service_name}"
    request = run_v2.GetServiceRequest(name=name)
    response = client.get_service(request=request)

    return response.uri


def get_prompt_conditional_router(
    prompt: str,
    num_models: int,
    routing_preference: RoutingPreference | None = None,
) -> RouterModule:
    from ypl.backend.llm.routing.prompt_router import RemotePromptCategorizerProposer

    endpoint = settings.PYTORCH_SERVE_GCP_URL
    key = settings.X_API_KEY
    preference = routing_preference or RoutingPreference(turns=[], user_id=None)

    reputable_proposer = RandomModelProposer(providers=set(settings.ROUTING_REPUTABLE_PROVIDERS))
    categorizer_proposer = RemotePromptCategorizerProposer(
        prompt,
        endpoint,
        key,
        exclude_unknown_models=False,
        skill_deficit_threshold=4,
    )

    if not preference.turns:
        # Construct a first-turn router guaranteeing at least two reputable models, focusing on speed but
        # also with random jitter.
        router: RouterModule = (
            (
                (ProModelProposer() | TopK(1)).with_flags(always_include=True, offset=100000)
                & (
                    reputable_proposer
                    | categorizer_proposer
                    | StreamableModelFilter()
                    | MaxSpeedProposer()
                    | RandomJitter(jitter_range=30.0)  # +/- 30 tokens per second
                    | ProviderFilter(one_per_provider=True)
                ).with_flags(always_include=True, offset=5000)
                & reputable_proposer.with_flags(offset=-1000, always_include=True)
            )
            | ProviderFilter(one_per_provider=True)
            | TopK(num_models)
            | RoutingDecisionLogger(
                enabled=settings.ROUTING_DO_LOGGING,
                prefix="first-prompt-conditional-router",
                metadata={"user_id": preference.user_id},
            )
        )
    else:
        # This is the router for all turns after the first; construct it based on the preference
        # and the branding. If both are bad, we default to one branded and one random. If one branded one is
        # good, we choose that and use a random model for the other.
        all_good_models = set()
        all_bad_models = set()

        for turn in preference.turns:
            if not turn.has_evaluation:
                continue

            for model in turn.models:
                if turn.preferred is None:
                    all_bad_models.add(model)
                else:
                    if model == turn.preferred:
                        all_good_models.add(model)
                    else:
                        all_bad_models.add(model)

        all_good_models = all_good_models - all_bad_models

        router: RouterModule = (  # type: ignore[no-redef]
            (
                (RandomModelProposer(models=all_good_models).with_flags(offset=100000) | TopK(1)).with_flags(
                    always_include=True
                )
                & (ProModelProposer() | Exclude(models=all_bad_models) | TopK(1)).with_flags(
                    always_include=True, offset=10000
                )
                & (
                    categorizer_proposer
                    | Exclude(models=all_bad_models)
                    | ProviderFilter(one_per_provider=True)
                    | MaxSpeedProposer()
                    | RandomJitter(jitter_range=30.0)  # +/- 30 tokens per second
                    | TopK(1)
                ).with_flags(always_include=True)
                & RandomModelProposer().with_flags(offset=-1000, always_include=True)
            )
            | Exclude(models=all_bad_models)
            | ProviderFilter(one_per_provider=True)
            | TopK(num_models)
            | RoutingDecisionLogger(
                enabled=settings.ROUTING_DO_LOGGING,
                prefix="nonfirst-prompt-conditional-router",
                metadata={
                    "user_id": preference.user_id,
                    "turns": [t.model_dump() for t in preference.turns],
                    "all_good_models": list(all_good_models),
                    "all_bad_models": list(all_bad_models),
                },
            )
        )

    return router


class HighErrorRateFilter(RNGMixin, ModelFilter):
    """
    Filters out models with a high error rate. If the error rate exceeds the soft threshold, the model is still
    considered for selection but only half the time. If the error rate exceeds the hard threshold, the model is
    excluded entirely for the duration of the time window.
    """

    def __init__(
        self,
        soft_threshold: float = 0.025,
        hard_threshold: float = 0.05,
        time_window: timedelta = timedelta(hours=6),
        soft_reject_prob: float = 0.5,
        min_count: int = 10,
    ):
        super().__init__(persist=True)
        self.soft_threshold = settings.ROUTING_ERROR_FILTER_SOFT_THRESHOLD or soft_threshold
        self.hard_threshold = settings.ROUTING_ERROR_FILTER_HARD_THRESHOLD or hard_threshold
        self.soft_reject_prob = settings.ROUTING_ERROR_FILTER_SOFT_REJECT_PROB or soft_reject_prob
        self.time_window = time_window
        self.min_count = min_count

    @cachetools.func.ttl_cache(ttl=60 * 5)  # 5 minutes
    def _get_error_rates(self) -> dict[str, float]:
        model_error_map: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

        with Session(get_engine()) as session:
            ret = session.exec(
                select(
                    LanguageModel.internal_name,
                    LanguageModelResponseStatus.status_type,
                    func.count(LanguageModelResponseStatus.status_type),  # type: ignore[arg-type]
                )
                .where(
                    LanguageModelResponseStatus.created_at > datetime.now() - self.time_window,  # type: ignore
                )
                .join(LanguageModel)
                .group_by(LanguageModel.internal_name, LanguageModelResponseStatus.status_type)
            ).all()

        for model_internal_name, status_type, error_count in ret:
            model_error_map[model_internal_name][status_type] = error_count

        return {
            model_internal_name: sum(
                v for c, v in error_counts.items() if not LanguageModelResponseStatusEnum(c).is_ok()
            )
            / sum(error_counts.values())
            for model_internal_name, error_counts in model_error_map.items()
            if sum(error_counts.values()) >= self.min_count
        }

    def _filter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        error_rates = self._get_error_rates()

        rejected_models = {
            model
            for model in state.selected_models
            if (error_rates.get(model, 0) > self.soft_threshold and self.get_rng().random() < self.soft_reject_prob)
            or error_rates.get(model, 0) > self.hard_threshold
        }

        return Exclude(models=rejected_models)._filter(state)


# Begin pro router logic and routine
ROUTING_LLM: BaseChatModel | None = None
ONLINE_LABELER = None
TOPIC_LABELER = None
MODIFIER_LABELER = None
USE_GEMINI_FOR_ROUTING = False


def get_routing_llm() -> BaseChatModel:
    global ROUTING_LLM
    if ROUTING_LLM is None:
        if USE_GEMINI_FOR_ROUTING:
            ROUTING_LLM = GeminiLangChainAdapter(
                model_info=ModelInfo(
                    provider=ChatProvider.GOOGLE,
                    model="gemini-2.0-flash-exp",
                    api_key=settings.GOOGLE_API_KEY,
                ),
                model_config_=dict(
                    project_id=settings.GCP_PROJECT_ID,
                    region=settings.GCP_REGION_GEMINI_2,
                    temperature=0.0,
                    max_output_tokens=32,
                    top_k=1,
                ),
            )
        else:
            ROUTING_LLM = OpenAILangChainAdapter(
                model_info=ModelInfo(
                    provider=ChatProvider.OPENAI,
                    model="gpt-4o-mini",
                    api_key=settings.OPENAI_API_KEY,
                ),
                model_config_=dict(
                    temperature=0.0,
                    max_tokens=40,
                ),
            )

    return ROUTING_LLM


def get_online_labeler() -> YuppOnlinePromptLabeler:
    global ONLINE_LABELER
    if ONLINE_LABELER is None:
        ONLINE_LABELER = YuppOnlinePromptLabeler(get_routing_llm(), timeout_secs=settings.ROUTING_TIMEOUT_SECS)
    return ONLINE_LABELER


def get_topic_labeler() -> YuppMultilabelClassifier:
    global TOPIC_LABELER
    if TOPIC_LABELER is None:
        TOPIC_LABELER = YuppMultilabelClassifier(get_routing_llm(), timeout_secs=settings.ROUTING_TIMEOUT_SECS)
    return TOPIC_LABELER


def get_modifier_labeler() -> PromptModifierLabeler:
    global MODIFIER_LABELER
    if MODIFIER_LABELER is None:
        MODIFIER_LABELER = PromptModifierLabeler(get_routing_llm(), timeout_secs=settings.ROUTING_TIMEOUT_SECS)
    return MODIFIER_LABELER


def get_good_and_bad_models(preference: RoutingPreference) -> tuple[set[str], set[str]]:
    """
    Go through all previous turns and split models into a good set and a bad set based on their user evaluation.
    """
    all_good_models = set()
    all_bad_models = set()
    # Go through all previous turns, add preferred models to all_good_models and others to all_bad_models
    for turn in preference.turns or []:
        if not turn.has_evaluation:
            continue
        if turn.preferred:
            all_good_models.add(turn.preferred)
            all_bad_models.update([m for m in turn.models if m != turn.preferred])
        else:
            all_bad_models.update(turn.models)
    all_good_models = all_good_models - all_bad_models
    return all_good_models, all_bad_models


async def get_simple_pro_router(
    prompt: str,
    num_models: int,
    routing_preference: RoutingPreference | None = None,
    reputable_providers: set[str] | None = None,
    user_selected_models: list[str] | None = None,
    show_me_more_models: list[str] | None = None,
    provided_categories: list[str] | None = None,
) -> RouterModule:
    from ypl.backend.llm.routing.rule_router import RoutingRuleFilter, RoutingRuleProposer

    preference = routing_preference or RoutingPreference(turns=[], user_id=None)
    reputable_proposer = RandomModelProposer(providers=reputable_providers or set(settings.ROUTING_REPUTABLE_PROVIDERS))
    online_labeler = get_online_labeler()
    topic_labeler = get_topic_labeler()
    modifier_labeler = get_modifier_labeler()
    online_label, topic_labels, modifier_labels = await asyncio.gather(
        online_labeler.alabel(prompt),
        topic_labeler.alabel(prompt),
        modifier_labeler.alabel(prompt),
    )

    online_category = "online" if online_label else "offline"
    categories = [online_category] + topic_labels + (provided_categories or [])
    categories = list(dict.fromkeys(categories))
    applicable_modifiers = modifier_labels

    rule_proposer = RoutingRuleProposer(*categories)
    rule_filter = RoutingRuleFilter(*categories)
    error_filter = HighErrorRateFilter()
    pro_proposer = ProModelProposer()
    num_pro = int(pro_proposer.get_rng().random() * 2 + 1)

    show_me_more_providers = (
        set(deduce_original_providers(tuple(show_me_more_models)).values()) if show_me_more_models else set()
    )

    def get_postprocessing_stage(exclude_models: set[str] | None = None) -> RouterModule:
        """
        Common post-processing stages that's shared in first-turn and non-first-turn routers
        """
        return (
            # -- filter stage --
            Exclude(providers=show_me_more_providers, models=exclude_models)
            # inject user selected model, even if they are already used before.
            # Also they are always treated with priority in the following dedup filters.
            | Inject(user_selected_models or [], score=10000000)
            | OnePerSemanticGroupFilter(priority_models=user_selected_models)  # dedupe by semantic group
            | ProviderFilter(one_per_provider=True, priority_models=user_selected_models)  # dedupe by provider
            # -- ranking stage --
            | TopK(num_models)  # keeps only top k models
            # -- annotation stage --
            | ModifierAnnotator(applicable_modifiers)  # annotate models with modifiers
            # -- logging stage --
            | RoutingDecisionLogger(
                enabled=settings.ROUTING_DO_LOGGING,
                prefix="first-prompt-simple-pro-router",
                metadata={
                    "user_id": preference.user_id,
                    "categories": categories,
                },
            )
        )

    if not preference.turns:
        # --- First Turn Router ---
        # Construct a first-turn router guaranteeing at least one pro model and one reputable model.
        router: RouterModule = (
            # -- candidate prep stage --
            rule_filter  # apply rules for an initial pass clean up on all candidate models, reject based on prompt
            # -- proposal stage --
            | (  # propose a set of models based on several heuristics
                (rule_proposer.with_flags(always_include=True) | RandomJitter(jitter_range=1))
                & (pro_proposer | error_filter | TopK(num_pro)).with_flags(always_include=True, offset=100000)
                & (StrongModelProposer() | error_filter | TopK(1)).with_flags(always_include=True, offset=50000)
                & (
                    reputable_proposer
                    | error_filter
                    | StreamableModelFilter()
                    | MaxSpeedProposer()
                    | RandomJitter(jitter_range=30.0)  # +/- 30 tokens per second
                    | ProviderFilter(one_per_provider=True)
                ).with_flags(always_include=True, offset=5000)
            )
            | error_filter  # removes models with high error rate
            # -- post processing stage --
            | get_postprocessing_stage(exclude_models=None)
        )
    else:
        # --- Non-First Turn Router ---
        all_good_models, all_bad_models = get_good_and_bad_models(preference)
        rule_filter.exempt_models = all_good_models

        router: RouterModule = (  # type: ignore[no-redef]
            # -- preprocessing stage --
            rule_filter
            # -- proposal stage --
            | (
                (RandomModelProposer(models=all_good_models) | error_filter | TopK(1)).with_flags(
                    always_include=True,
                    offset=10000000,
                )
                & (rule_proposer.with_flags(always_include=True) | error_filter | RandomJitter(jitter_range=1))
                & (
                    (ProModelProposer() | Exclude(models=all_bad_models) | error_filter | TopK(1)).with_flags(
                        always_include=True, offset=10000
                    )
                    ^ (
                        reputable_proposer
                        | StreamableModelFilter()
                        | error_filter
                        | MaxSpeedProposer()
                        | RandomJitter(jitter_range=30.0)  # +/- 30 tokens per second
                        | ProviderFilter(one_per_provider=True)
                    ).with_flags(always_include=True, offset=10000)
                    ^ RandomModelProposer().with_flags(offset=10000, always_include=True)
                ).with_probs(
                    settings.ROUTING_WEIGHTS.get("pro", 0.5),
                    settings.ROUTING_WEIGHTS.get("reputable", 0.25),
                    settings.ROUTING_WEIGHTS.get("random", 0.25),
                )
                & RandomModelProposer().with_flags(offset=-1000, always_include=True)
            )
            | error_filter  # removes models with high error rate
            # -- post processing stage --
            | get_postprocessing_stage(exclude_models=all_bad_models)
        )

    return router
