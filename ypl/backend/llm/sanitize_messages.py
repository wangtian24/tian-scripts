import logging

from langchain_core.messages import AIMessage, BaseMessage

from ypl.backend.llm.model_heuristics import ModelHeuristics

DEFAULT_MAX_TOKENS: int = 32_000

model_heuristics = ModelHeuristics(tokenizer_type="tiktoken")


def sanitize_messages(
    messages: list[BaseMessage], system_prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS
) -> list[BaseMessage]:
    """Sanitizes the list of messages to send to a model, drop any entries that can potentially cause an error."""
    return truncate_message(replace_empty_messages(messages), system_prompt, max_tokens)


# TODO(bhanu, gilad) to revisit truncation with alternates mentioned here - https://github.com/yupp-ai/yupp-mind/pull/633#discussion_r1896057154
def truncate_message(
    messages: list[BaseMessage], system_prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS
) -> list[BaseMessage]:
    """
    Truncates messages to a maximum number of tokens.

    The messages towards the end of the list will be preserved first. Internally this leaves
    5% buffer to handle markup that may be added by various AI SDKs.

    Args:
        messages: The messages to truncate
        system_prompt: The system prompt to consider in token count
        max_tokens: The maximum number of tokens to allow, or None for default limit

    Returns:
        The truncated messages
    """

    # leave a 5% buffer for additional markup
    available_tokens = max_tokens * 0.95 - len(model_heuristics.encode_tokens(system_prompt))
    truncated_messages: list[BaseMessage] = []

    # Iterate through messages in reverse order
    for message in reversed(messages):
        if not isinstance(message.content, str):
            truncated_messages.insert(0, message)
            continue
        message_tokens = model_heuristics.encode_tokens(message.content)
        message_token_length = len(message_tokens)

        if available_tokens - message_token_length >= 0:
            truncated_messages.insert(0, message)
            available_tokens -= message_token_length
        else:
            # If the message doesn't fit entirely, truncate it
            truncated_content = model_heuristics.decode_tokens(message_tokens[-int(available_tokens) :])

            # Modify the message directly instead of creating a copy
            message.content = "[some of the message was too long, so it was truncated...] " + truncated_content

            truncated_messages.insert(0, message)
            break

    return truncated_messages


def replace_empty_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Remove any empty messages.

    Most model APIs (claude, gemini etc) will consider a CoreMessage with no content
    to be an invalid argument. Messages may also be None due to a run time error.
    """
    filtered_messages = []

    for index, message in enumerate(messages):
        if message is None:
            logging.warn(
                f"Encountered an undefined message at index {index} in the message history. "
                "Dropping the entry, but this could lead to model errors since models may "
                "expect alternating user and assistant messages."
            )
            continue

        if isinstance(message, AIMessage) and message.content == "":
            message.content = "<no response>"

        filtered_messages.append(message)

    return filtered_messages
