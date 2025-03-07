from langchain_core.language_models.chat_models import BaseChatModel

from ypl.backend.config import settings
from ypl.backend.llm.judge import PromptModifierLabeler
from ypl.backend.llm.provider.provider_clients import get_provider_client
from ypl.backend.llm.routing.router import get_default_routing_llm

MODIFIER_LABELER: PromptModifierLabeler | None = None

PROMPT_MODIFIER_MODEL = "gemini-1.5-flash-8b"


async def get_prompt_modifiers(
    prompt: str,
) -> list[str]:
    """
    Get the prompt modifier from the prompt
    """
    modifier_labeler = await _get_modifier_labeler()
    return await modifier_labeler.alabel(prompt)


async def _get_modifier_labeler() -> PromptModifierLabeler:
    global MODIFIER_LABELER
    if MODIFIER_LABELER is None:
        MODIFIER_LABELER = PromptModifierLabeler(
            await get_prompt_modifier_llm(PROMPT_MODIFIER_MODEL), timeout_secs=settings.ROUTING_TIMEOUT_SECS
        )
    return MODIFIER_LABELER


async def get_prompt_modifier_llm(model_name: str | None = None) -> BaseChatModel:
    if model_name:
        return await get_provider_client(internal_name=model_name)
    else:
        return await get_default_routing_llm()
