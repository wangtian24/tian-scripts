from ypl.backend.config import settings
from ypl.backend.llm.judge import PromptModifierLabeler
from ypl.backend.llm.routing.modules.base import RouterModule
from ypl.backend.llm.routing.router import _get_routing_llm
from ypl.backend.llm.routing.router_state import RouterState

MODIFIER_LABELER: PromptModifierLabeler | None = None


class ModifierAnnotator(RouterModule):
    """Just adds the applicable modifiers to the state."""

    def __init__(self, modifiers: list[str]) -> None:
        self.modifiers = modifiers

    def _select_models(self, state: RouterState) -> RouterState:
        return state.emplaced(applicable_modifiers=self.modifiers)

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return state.emplaced(applicable_modifiers=self.modifiers)


async def run_prompt_modifier_on_models(
    prompt: str,
    selected_models: list[str],
) -> RouterState:
    """
    Get the prompt modifier results for the selected models, run this as a one-step router chain.
    """
    modifier_labeler = _get_modifier_labeler()
    modifier_labels = await modifier_labeler.alabel(prompt)
    router: RouterModule = ModifierAnnotator(modifier_labels)

    start_state = RouterState.new_chosen_models_state(selected_models)
    return router.select_models(state=start_state)


def _get_modifier_labeler() -> PromptModifierLabeler:
    global MODIFIER_LABELER
    if MODIFIER_LABELER is None:
        MODIFIER_LABELER = PromptModifierLabeler(_get_routing_llm(), timeout_secs=settings.ROUTING_TIMEOUT_SECS)
    return MODIFIER_LABELER
