from langchain_core.language_models.chat_models import BaseChatModel

from ypl.backend.config import settings
from ypl.backend.llm.judge import PromptModifierLabeler
from ypl.backend.llm.provider.provider_clients import get_provider_client
from ypl.backend.llm.routing.modules.base import RouterModule
from ypl.backend.llm.routing.router import _get_routing_llm
from ypl.backend.llm.routing.router_state import RouterState

MODIFIER_LABELER: PromptModifierLabeler | None = None

PROMPT_MODIFIER_MODEL = "gemini-1.5-flash-8b"


class ModifierAnnotator(RouterModule):
    """Just adds the applicable modifiers to the state."""

    def __init__(self, modifiers: list[str]) -> None:
        self.modifiers = modifiers

    def _select_models(self, state: RouterState) -> RouterState:
        return state.emplaced(applicable_modifiers=self.modifiers)

    async def _aselect_models(self, state: RouterState) -> RouterState:
        return state.emplaced(applicable_modifiers=self.modifiers)


async def get_prompt_modifiers(
    prompt: str,
) -> list[str]:
    """
    Get the prompt modifier from the prompt
    """
    modifier_labeler = await _get_modifier_labeler()
    return await modifier_labeler.alabel(prompt)


async def attach_prompt_modifiers_to_models(
    modifier_labels: list[str],
    selected_models: list[str],
) -> RouterState:
    """
    Attach prompt modifiers to selected models, run a one-step router chain.
    """
    router: RouterModule = ModifierAnnotator(modifier_labels)
    start_state = RouterState.new_chosen_models_state(selected_models)
    return router.select_models(state=start_state)


async def _get_modifier_labeler() -> PromptModifierLabeler:
    global MODIFIER_LABELER
    if MODIFIER_LABELER is None:
        MODIFIER_LABELER = PromptModifierLabeler(
            await get_prompt_modifier_llm(PROMPT_MODIFIER_MODEL), timeout_secs=settings.ROUTING_TIMEOUT_SECS
        )
    return MODIFIER_LABELER


async def get_prompt_modifier_llm(model_name: str | None = None) -> BaseChatModel:
    if model_name:
        return await get_provider_client(model_name)
    else:
        return _get_routing_llm()
