import uuid

from cachetools.func import ttl_cache
from langchain_core.prompts import ChatPromptTemplate
from sqlmodel import Session, select

from ypl.backend.db import get_engine
from ypl.backend.prompts.system_prompts import get_system_prompt
from ypl.db.chats import PromptModifier

PROMPT_MODIFIER_PREAMBLE = """
Here are additional instructions regarding style, conciseness, and formatting;
only adhere to them if they don't contradict the user's prompt:
"""

ALL_MODELS_IN_CHAT_HISTORY_PROMPT = """
The user in this conversation has access to additional AI models that respond to the same prompts.
Responses from these models to previous prompts in the conversation are included along with your responses, and annotated
with comments such as:
- "This was your response:"
- "A response from another assistant:"
- "(Your response was empty)"
- "(This response was preferred by the user)"
You may refer to these past responses, but do not repeat them, and never include annotations such as
"A response from another assistant:" and the other ones above in your response.
"""


@ttl_cache(ttl=600)  # 600 seconds = 10 minutes
def load_prompt_modifiers() -> dict[uuid.UUID, PromptModifier]:
    """Load all prompt modifiers from the database.
        Results are cached for configured ttl.

    Returns:
        Dictionary mapping prompt_modifier_id to PromptModifier objects
    """
    # Query all prompt modifiers
    stmt = select(PromptModifier)
    with Session(get_engine()) as session:
        prompt_modifiers = session.exec(stmt).all()

    # Create dictionary mapping id to modifier
    return {modifier.prompt_modifier_id: modifier for modifier in prompt_modifiers}


def get_prompt_modifiers(prompt_modifier_ids: list[uuid.UUID]) -> str:
    """
    Each modifier's text is concatenated with newlines if the ID exists.

    Args:
        prompt_modifier_ids: List of prompt modifier IDs to include

    Returns:
        String containing concatenated modifier texts
    """
    # Load all modifiers from DB
    all_modifiers = load_prompt_modifiers()

    # Get text for each valid modifier ID
    return "\n".join(
        all_modifiers[modifier_id].text for modifier_id in prompt_modifier_ids if modifier_id in all_modifiers
    )


def get_system_prompt_with_modifiers(
    model: str, modifier_ids: list[uuid.UUID] | None, use_all_models_in_chat_history: bool
) -> str:
    """
    Get the system prompt for a model with any additional prompt modifiers applied.

    Args:
        model: The name of the model to get the system prompt for
        modifier_ids: List of prompt modifier IDs to include in the system prompt

    Returns:
        The complete system prompt string with any modifiers appended
    """

    # o1-preview and o1-mini models don't support modifiers. o1 is fine though
    if model.startswith(("o1-preview", "o1-mini")):
        return ""

    base_system_prompt = get_system_prompt(model)
    if use_all_models_in_chat_history:
        base_system_prompt += ALL_MODELS_IN_CHAT_HISTORY_PROMPT

    is_image_generation_model = "dall-e" in model
    prompt_modifiers = get_prompt_modifiers(modifier_ids) if modifier_ids and not is_image_generation_model else ""

    if not prompt_modifiers:
        return base_system_prompt

    return base_system_prompt + "\n" + PROMPT_MODIFIER_PREAMBLE + prompt_modifiers


JUDGE_PROMPT_MODIFIER_PROMPT_TEMPLATE = """
You are an AI assistant specialized in determining which modifications can be provided as instructions to another AI assistant
to change the style of a response to a prompt.

The following are the prompt modifiers available, and the instructions that would be provided to another AI assistant for each:

{modifiers}

Here is a user's prompt. Respond with a comma-separated list of modifier names that could be applied to the prompt.
Do not wrap the list in any markup.

User's message:
{prompt}
"""

JUDGE_PROMPT_MODIFIER_PROMPT = ChatPromptTemplate.from_messages([("human", JUDGE_PROMPT_MODIFIER_PROMPT_TEMPLATE)])
