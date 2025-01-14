import logging
from typing import Any

from ypl import __version__
from ypl.backend.llm.routing.debug import RoutingDebugInfo, build_routing_debug_info
from ypl.backend.llm.routing.modules.base import RouterModule
from ypl.backend.llm.routing.route_data_type import RoutingPreference
from ypl.backend.llm.routing.router_state import RouterState
from ypl.backend.utils.json import json_dumps


class RoutingDecisionLogger(RouterModule):
    def __init__(
        self,
        enabled: bool = True,
        prefix: str = "router",
        preference: RoutingPreference | None = None,
        required_models: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:  # noqa: E501
        """
        Args:
            enabled: Whether to log the routing decision.
        """
        self.enabled = enabled
        self.prefix = prefix
        self.metadata = metadata or {}
        self.preference = preference
        self.required_models = required_models

    def _select_models(self, state: RouterState) -> RouterState:
        """
        Log the routing decision.
        """
        if self.enabled:
            criteria = [
                (str(criteria.name), score)
                for _, criteria_map in state.selected_models.items()
                for criteria, score in criteria_map.items()
            ]

            # add criteria to model debug
            for model, debug_info in state.model_journey.items():
                if model in state.selected_models:
                    state.model_journey[model] = f"{debug_info} {criteria}"

            # create full routing debug info
            routing_debug_info: RoutingDebugInfo = build_routing_debug_info(
                selected_models_rs=state,
                fallback_models_rs=None,
                required_models=self.required_models,
            )

            log_dict = {
                "message": f"Model routing decision [{self.prefix}]: {', '.join(state.selected_models.keys())}",
                "codebase_version": __version__,
                "prefix": self.prefix,
                "chosen_model_names": list(state.selected_models.keys()),
                "selection_criteria": criteria,
                "applicable_prompt_modifiers": state.applicable_modifiers,
                "routing_debug_info": routing_debug_info,
                "additional_metadata": self.metadata,
            }
            logging.info(json_dumps(log_dict))
            print("---------------logging decision---------------")

        return state
