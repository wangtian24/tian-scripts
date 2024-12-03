import re
from collections import defaultdict
from copy import deepcopy

from cachetools.func import ttl_cache
from sqlmodel import Session, select

from ypl.backend.db import get_engine
from ypl.backend.llm.chat import deduce_original_providers, standardize_provider_name
from ypl.backend.llm.routing.policy import SelectionCriteria
from ypl.backend.llm.routing.router import ModelFilter, ModelProposer, RouterState
from ypl.backend.llm.utils import dict_shuffled
from ypl.db.language_models import RoutingAction, RoutingRule
from ypl.utils import RNGMixin


class RoutingTable(RNGMixin):
    def __init__(self, rules: list[RoutingRule]):
        self.rules_cat_map: defaultdict[str, list[RoutingRule]] = defaultdict(list)

        for rule in rules:
            self.rules_cat_map[rule.source_category].append(rule)

    def apply(self, category: str, models: set[str]) -> tuple[dict[str, float], set[str]]:
        """
        Apply the routing rules to the given models and return a dictionary of model scores. The score is the sum of
        the z-indices of the rules that matched the model. If none of the rules match, the score is 0.

        Args:
            category: The category of the prompt. "*" matches all categories.
            models: The models to apply the rules to.

        Returns:
            A tuple of a dictionary of model scores and a set of rejected models.
        """
        candidate_models = deepcopy(models)
        provider_map = deduce_original_providers(tuple(candidate_models))

        accept_reject_map: dict[RoutingAction, set[tuple[str, float]]] = {
            RoutingAction.ACCEPT: set(),
            RoutingAction.REJECT: set(),
        }

        all_rules = deepcopy(self.rules_cat_map["*"])

        if category != "*":
            all_rules.extend(self.rules_cat_map[category])

        all_rules.sort(key=lambda x: x.z_index, reverse=True)

        for rule in all_rules:
            if rule.target.noop():
                continue

            for model in list(candidate_models):
                provider = provider_map[model]
                matched = False

                if rule.destination == "*":
                    matched = True
                else:
                    dest_provider, dest_model = rule.destination.split("/", 1)
                    provider = standardize_provider_name(provider)
                    dest_provider = standardize_provider_name(dest_provider)
                    dest_model_pattern = rf"^{dest_model.replace('*', '.+')}$"

                    if provider == dest_provider and (dest_model == "*" or re.match(dest_model_pattern, model)):
                        matched = True

                if (
                    matched
                    and model not in accept_reject_map[rule.target]
                    and model not in accept_reject_map[rule.target.opposite()]
                    and self.get_rng().random() <= rule.probability
                ):
                    accept_reject_map[rule.target].add((model, rule.z_index))
                    candidate_models.remove(model)

        return {
            **dict_shuffled(
                {model: float(z_index) for model, z_index in accept_reject_map[RoutingAction.ACCEPT]},
                self.get_rng(),
            ),
            **dict_shuffled(
                {model: 0.0 for model in candidate_models},
                self.get_rng(),
            ),
        }, {model for model, _ in accept_reject_map[RoutingAction.REJECT]}


@ttl_cache(ttl=3600)  # Cache for 1 hour
def get_routing_table() -> RoutingTable:
    with Session(get_engine()) as session:
        return RoutingTable(list(session.exec(select(RoutingRule).where(RoutingRule.is_active)).all()))


class RoutingRuleProposer(ModelProposer):
    def __init__(self, prompt_category: str) -> None:
        self.prompt_category = prompt_category
        self.routing_table = get_routing_table()

    def _propose_models(self, models_to_select: set[str], state: RouterState) -> RouterState:
        selected_models, _ = self.routing_table.apply(self.prompt_category, models_to_select)

        return state.emplaced(
            selected_models={
                model: {SelectionCriteria.ROUTING_RULES: score} for model, score in selected_models.items()
            }
        )


class RoutingRuleFilter(ModelFilter):
    def __init__(self, prompt_category: str) -> None:
        super().__init__(persist=True)
        self.prompt_category = prompt_category
        self.routing_table = get_routing_table()

    def _filter(self, state: RouterState) -> tuple[RouterState, set[str]]:
        _, rejected_models = self.routing_table.apply(self.prompt_category, set(state.selected_models.keys()))
        return state.emplaced(excluded_models=rejected_models), rejected_models
