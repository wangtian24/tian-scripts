import logging
from collections.abc import Sequence
from typing import Any

from ypl.backend.llm.routing.modules.base import RouterModule
from ypl.backend.llm.routing.router_state import RouterState
from ypl.backend.utils.json import json_dumps


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
            "message": f"Model routing decision: {', '.join(self.chosen_model_names)}",
            "codebase_version": self.codebase_version,
            "prefix": self.prefix,
            "candidate_model_names": self.candidate_model_names,
            "chosen_model_names": self.chosen_model_names,
            "selection_criteria": self.selection_criteria,
            "applicable_prompt_modifiers": self.applicable_prompt_modifiers,
            "additional_metadata": self.additional_metadata,
        }

        logging.info(json_dumps(log_dict))


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
