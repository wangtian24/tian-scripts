from abc import abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta

import cachetools
from sqlalchemy import func
from sqlmodel import Session, select
from ypl.backend.config import settings
from ypl.backend.db import get_engine
from ypl.backend.llm.constants import MODEL_HEURISTICS
from ypl.backend.llm.db_helpers import (
    deduce_original_providers,
    deduce_semantic_groups,
    get_active_models,
    get_image_attachment_models,
    get_model_context_lengths,
)
from ypl.backend.llm.model_heuristics import ModelHeuristics
from ypl.backend.llm.routing.modules.base import RouterModule
from ypl.backend.llm.routing.policy import SelectionCriteria
from ypl.backend.llm.routing.router_state import RouterState
from ypl.backend.prompts import FALLBACK_SYSTEM_PROMPT
from ypl.db.language_models import LanguageModel, LanguageModelResponseStatus, LanguageModelResponseStatusEnum
from ypl.utils import RNGMixin


class ModelFilter(RouterModule):
    """
    Represents a filter of a set of models to route to. Subclasses should implement the `_filter` method
    to define the specific model filtering logic.
    """

    def __init__(self, name: str, persist: bool = False, exempt_models: set[str] | None = None) -> None:
        """
        Args:
            name: The name of the filter, will be attached to the model_journey_debug for the affected models.
            persist: Whether to persist the models that are excluded from the selection, so that they are never
                selected again in future routing modules.
            exempt_models: The models to exempt from the filter.
        """
        self.name = name
        self.persist = persist
        self.exempt_models = exempt_models or set()

    def _select_models(self, state: RouterState) -> RouterState:
        state, excluded_models = self._filter(state)
        return self._process_response(state, excluded_models)

    async def _aselect_models(self, state: RouterState) -> RouterState:
        state, excluded_models = await self._afilter(state)
        return self._process_response(state, excluded_models)

    def _process_response(self, state: RouterState, excluded_models: set[str]) -> RouterState:
        excluded_models = excluded_models - self.exempt_models
        state.excluded_models = state.excluded_models - self.exempt_models

        if self.persist:
            state.excluded_models = state.excluded_models.union(excluded_models)

        for model in excluded_models:
            state.model_journey[model] += f" {self.name}"

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
        super().__init__(name=f"-topK({k})", persist=persist)
        self.k = k

    def _filter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        selected_models = sorted(
            state.selected_models.items(),  # (str, dict[SelectionCriteria, float])
            key=lambda x: sum(x[1].values()),  # sum of all scores from all criteria
            reverse=True,
        )
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
        super().__init__(name="*jitter", persist=persist)
        self.jitter_range = jitter_range
        self.jitter_pct = jitter_pct

    def _filter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        for model, scores in state.selected_models.items():
            if self.jitter_range is not None:
                jitter = self.get_rng().uniform(-self.jitter_range, self.jitter_range) / len(scores)
            elif self.jitter_pct is not None:
                jitter = self.get_rng().uniform(-self.jitter_pct, self.jitter_pct)
                jitter = jitter * sum(scores.values()) / len(scores)
            else:
                raise ValueError("Either jitter_range or jitter_pct must be provided")

            for criterion, score in scores.items():
                scores[criterion] = max(0, score + jitter)

            # update debug info
            if model in state.model_journey:
                state.model_journey[model] = f"{state.model_journey[model]} jitter({jitter:.2f})"
            else:
                state.model_journey[model] = f"jitter({jitter:.2f})"

        return state, set()


class Inject(ModelFilter):
    def __init__(self, models: list[str], *, score: float = 1) -> None:
        super().__init__(name="+inject", persist=True)
        self.models = models
        self.score = score

    def _filter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        for m in self.models:
            if m in state.model_journey:
                state.model_journey[m] = f"{state.model_journey[m]} +inject({self.score:.2f})"
            else:
                state.model_journey[m] = f"inject({self.score:.2f})"

        return state.emplaced(
            selected_models={
                **state.selected_models,
                **{model: {SelectionCriteria.INJECT: self.score - idx} for idx, model in enumerate(self.models)},
            },
        ), set()


class Exclude(ModelFilter):
    """
    Represents a filter to remove certain models and models from certain providers.
    """

    def __init__(self, *, name: str | None = None, models: set[str] | None = None, providers: set[str] | None = None):
        super().__init__(name=name or "-exclude", persist=True)
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
        super().__init__(name="-nonStreamable", models=non_streaming_models)


class SupportsImageAttachmentModelFilter(Exclude):
    """Filter to models that support image attachments."""

    def __init__(self) -> None:
        non_image_attachment_models = set(get_active_models()) - set(get_image_attachment_models())
        super().__init__(name="-noImageAttachment", models=non_image_attachment_models)


class ContextLengthFilter(Exclude):
    """Filter models based on their context length."""

    def __init__(self, prompt: str, max_length_fraction: float = 0.8) -> None:
        """
        Filters models whose context length is insufficient for the prompt.
        Args:
            prompt: The prompt to filter models by.
            max_length_fraction: The maximum fraction of the context length to allow.
        """
        # We don't know the specific system prompt because routing hasn't happened yet, so we use a default one.
        prompt = FALLBACK_SYSTEM_PROMPT + " " + prompt
        prompt_length = len(ModelHeuristics(tokenizer_type="tiktoken").encode_tokens(prompt))
        context_lengths = get_model_context_lengths()
        excluded_models = {
            model
            for model, context_length in context_lengths.items()
            if context_length < prompt_length * max_length_fraction
        }
        super().__init__(name="-contextLength", models=excluded_models)


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
        super().__init__(name="-provider", persist=persist)

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
        super().__init__(name="-semanticGroup", persist=persist)
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
        super().__init__(name="-highErrorRate", persist=True)
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

        return Exclude(name="-highErrorRate", models=rejected_models)._filter(state)
