import logging
import uuid
from collections import defaultdict
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from sqlalchemy.orm import joinedload
from sqlmodel import Session, or_, select

from ypl.backend.db import get_engine
from ypl.backend.llm.chat import ALL_MODELS_IN_CHAT_HISTORY_PREAMBLE, MAX_LOGGED_MESSAGE_LENGTH, RESPONSE_SEPARATOR
from ypl.backend.llm.db_helpers import is_image_generation_model
from ypl.backend.utils.json import json_dumps
from ypl.db.attachments import Attachment
from ypl.db.chats import Chat, ChatMessage, CompletionStatus, MessageType, MessageUIStatus, Turn
from ypl.db.language_models import LanguageModel


class ChatContext(BaseModel):
    """Represents the chat context with formatted messages and optional current turn responses.

    Attributes:
        messages: List of formatted messages (HumanMessage, AIMessage, SystemMessage) representing the chat history.
        uuids: List of UUIDs corresponding to the messages.
        current_turn_responses: Optional mapping of model names to their corresponding ChatMessage for the current turn.
        current_turn_models: Optional mapping of model names to their corresponding model labels for the current turn.
        current_turn_quicktake: Optional quicktake for the current turn.
    """

    uuids: list[uuid.UUID | None]
    messages: list[BaseMessage]
    current_turn_responses: dict[str, ChatMessage] | None = None
    current_turn_models: dict[str, str] | None = None
    current_turn_quicktake: str | None = None


def _get_enhanced_user_message(
    messages: list[ChatMessage], max_message_length: int | None = None
) -> tuple[UUID, HumanMessage]:
    user_msgs = [msg for msg in messages if msg.message_type == MessageType.USER_MESSAGE]
    if not user_msgs:
        raise ValueError("No user messages found")
    if len(user_msgs) > 1:
        raise ValueError("Multiple user messages found")
    user_msg = user_msgs[0]
    attachments = user_msg.attachments or []
    content = (
        user_msg.content[:max_message_length] + "..."
        if max_message_length and len(user_msg.content) > max_message_length
        else user_msg.content
    )
    return (
        user_msg.message_id,
        HumanMessage(
            content=content,
            additional_kwargs={"attachments": attachments},
        ),
    )


def _get_assistant_messages(
    turn_messages: list[ChatMessage],
    model: str,
    use_all_models_in_chat_history: bool,
    max_message_length: int | None = None,
) -> list[BaseMessage]:
    """Get assistant messages for a turn.

    If use_all_models_in_chat_history is True, includes assistant messages from all models, indicating which ones
    are from the current model and which one was preferred by the user (if any).
    If use_all_models_in_chat_history is False, includes only the preferred messages, or the first message if none
    were selected.

    This is used in test only right now.
    """
    messages: list[BaseMessage] = []
    assistant_msgs = [
        msg
        for msg in turn_messages
        if msg.message_type == MessageType.ASSISTANT_MESSAGE and msg.content and msg.assistant_language_model
    ]
    if not assistant_msgs:
        return messages

    if use_all_models_in_chat_history and len(assistant_msgs) > 1:
        all_content = []
        for msg in assistant_msgs:
            content = msg.content or ""
            if max_message_length and len(content) > max_message_length:
                content = content[:max_message_length] + "..."
            if msg.assistant_language_model.internal_name == model:
                # A previous response from the current assistant.
                if content:
                    content = "This was your response:\n\n" + content
            else:
                # A previous response from another assistant.
                if content:
                    # Only include responses from other assistants if non-empty.
                    content = "A response from another assistant:\n\n" + content
            if msg.ui_status == MessageUIStatus.SELECTED and content:
                content += "\n\n(This response was preferred by the user)"
            if content:
                all_content.append(content)
        if all_content:
            content = ""
            if model:
                content += ALL_MODELS_IN_CHAT_HISTORY_PREAMBLE
            content += RESPONSE_SEPARATOR.join(all_content)
            messages.append(AIMessage(content=content))
    else:
        selected_msg = next(
            (msg for msg in assistant_msgs if msg.ui_status == MessageUIStatus.SELECTED),
            None,
        )
        if not selected_msg:
            selected_msg = next(
                (msg for msg in assistant_msgs if msg.assistant_language_model.internal_name == model),
                assistant_msgs[0],  # Fallback to first message if none selected
            )
        if selected_msg:
            content = selected_msg.content
        else:
            content = None
            log_info = {
                "message": "No selected message in turn",
                "model_for_selected_message_lookup": model,
                "turn_id": assistant_msgs[0].turn_id,
            }
            logging.warning(json_dumps(log_info))

        # if content is null, a place holder is added as part of sanitize_messages.py/replace_empty_messages()
        messages.append(AIMessage(content=content))

    return messages


async def get_curated_chat_context(
    chat_id: UUID,
    use_all_models_in_chat_history: bool,
    model: str,
    current_turn_id: UUID | None = None,
    include_current_turn: bool = False,
    max_turns: int = 20,
    max_message_length: int | None = None,
    context_for_logging: str | None = None,
    return_all_current_turn_responses: bool = False,
) -> ChatContext:
    """Fetch chat history and format it for OpenAI context, returning message UUIDs as well.

    Note: Non-human chat messages will not return UUIDs because multiple messages may be
    merged into a single message by this method.

    Args:
        chat_id: The chat ID to fetch history for.
        use_all_models_in_chat_history: Whether to include all models in the chat history.
        model: The model to fetch history for.
        current_turn_id: The current turn ID.
        include_current_turn: Whether to include the current turn and all turns up to it in the chat history.
                            When False, excludes the current turn and any turns after it.
        max_turns: Maximum number of turns to include in history.
        max_message_length: Maximum length of each message.
        context_for_logging: Context string for logging.
        return_all_current_turn_responses: If True and include_current_turn is True, returns all model
            responses for the current turn separately as a dictionary in ChatContext.current_turn_responses.

    Returns:
        ChatContext containing formatted messages and optionally current turn responses and quicktake.
    """
    assert not (
        return_all_current_turn_responses and not include_current_turn
    ), "Cannot return all current turn responses if current turn is not included"
    query = (
        select(ChatMessage)
        .join(Turn, Turn.turn_id == ChatMessage.turn_id)  # type: ignore[arg-type]
        .join(Chat, Chat.chat_id == Turn.chat_id)  # type: ignore[arg-type]
        .outerjoin(Attachment, Attachment.chat_message_id == ChatMessage.message_id)  # type: ignore[arg-type]
        .options(
            joinedload(ChatMessage.assistant_language_model).load_only(  # type: ignore[arg-type]
                LanguageModel.internal_name,  # type: ignore[arg-type]
                LanguageModel.label,  # type: ignore[arg-type]
            ),
            joinedload(ChatMessage.attachments),  # type: ignore
        )
        .where(
            Chat.chat_id == chat_id,
            ChatMessage.deleted_at.is_(None),  # type: ignore[union-attr]
            Turn.deleted_at.is_(None),  # type: ignore[union-attr]
            or_(
                # Do not include errored responses.
                ChatMessage.completion_status == CompletionStatus.SUCCESS,
                ChatMessage.completion_status.is_(None),  # type: ignore[attr-defined]
            ),
            Chat.deleted_at.is_(None),  # type: ignore[union-attr]
            Turn.turn_id.in_(  # type: ignore[attr-defined]
                select(Turn.turn_id)
                .where(Turn.chat_id == chat_id)
                .order_by(Turn.sequence_id.desc())  # type: ignore[attr-defined]
                .limit(max_turns)
            ),
        )
        .order_by(
            Turn.sequence_id.asc(),  # type: ignore[attr-defined]
            ChatMessage.turn_sequence_number.asc(),  # type: ignore[union-attr]
        )
    )

    if current_turn_id:
        subquery = select(Turn.sequence_id).where(Turn.turn_id == current_turn_id).scalar_subquery()
        if not include_current_turn:
            query = query.where(Turn.sequence_id < subquery)
        else:
            query = query.where(Turn.sequence_id <= subquery)

    formatted_messages: list[tuple[UUID | None, BaseMessage]] = []
    # An async session is 2-3X slower.
    with Session(get_engine()) as session:
        result = session.exec(query)
        # Limit to the most recent messages.
        messages = result.unique().all()

    # Group messages by turn_id
    turns: defaultdict[UUID, list[ChatMessage]] = defaultdict(list)
    for msg in messages:
        turns[msg.turn_id].append(msg)

    is_image_gen = await is_image_generation_model(model)
    # The loop below proceeds in insertion order, which is critical for
    # the correctness of this method.
    for turn_messages in turns.values():
        # Get user messages
        formatted_messages.append(_get_enhanced_user_message(turn_messages, max_message_length))
        if not is_image_gen:
            # Get assistant messages only if it's not for the image-generation model.
            formatted_messages.extend(
                [
                    (None, x)
                    for x in _get_assistant_messages(
                        turn_messages, model, use_all_models_in_chat_history, max_message_length
                    )
                ]
            )

    info = {
        "message": f"chat_context ({context_for_logging or 'no context'})",
        "chat_id": str(chat_id),
        "model": model,
    }
    for i, (_unused_uuid, fmsg) in enumerate(formatted_messages):
        msg_type = (
            "Human"
            if isinstance(fmsg, HumanMessage)
            else "AI"
            if isinstance(fmsg, AIMessage)
            else "Sys"
            if isinstance(fmsg, SystemMessage)
            else type(fmsg).__name__
        )
        info[f"m{i}_{msg_type}"] = (
            str(fmsg.content[:MAX_LOGGED_MESSAGE_LENGTH]) + "..."
            if len(fmsg.content) > MAX_LOGGED_MESSAGE_LENGTH
            else str(fmsg.content)
        )
    logging.info(json_dumps(info))

    if return_all_current_turn_responses:
        current_turn_responses = {}
        current_turn_models = {}
        current_turn_quicktake = None
        for m in messages[::-1]:
            if (
                m.message_type == MessageType.ASSISTANT_MESSAGE
                and m.turn_id == current_turn_id
                and m.assistant_model_name
                and m.completion_status in (CompletionStatus.SUCCESS, CompletionStatus.USER_ABORTED)
            ):
                current_turn_responses[m.assistant_model_name] = m
                current_turn_models[m.assistant_model_name] = str(m.assistant_language_model.label)
            elif (
                m.message_type == MessageType.QUICK_RESPONSE_MESSAGE
                and m.turn_id == current_turn_id
                and m.ui_status != MessageUIStatus.OBSOLETE
            ):
                current_turn_quicktake = m.content
            else:
                break
        return ChatContext(
            messages=[msg for _, msg in formatted_messages],
            uuids=[msg_uuid for msg_uuid, _ in formatted_messages],
            current_turn_responses=current_turn_responses,
            current_turn_models=current_turn_models,
            current_turn_quicktake=current_turn_quicktake,
        )

    return ChatContext(
        messages=[msg for _, msg in formatted_messages], uuids=[msg_uuid for msg_uuid, _ in formatted_messages]
    )
