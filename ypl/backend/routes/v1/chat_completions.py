import asyncio  # noqa: I001
import json
import logging
import traceback
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from ypl.backend.config import settings
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.db import get_async_engine
from ypl.backend.llm.chat import Intent, check_for_stop_request, get_curated_chat_context, upsert_chat_message
from ypl.backend.llm.model.model import ModelResponseTelemetry
from ypl.backend.llm.provider.provider_clients import get_language_model, get_provider_client
from ypl.backend.llm.sanitize_messages import DEFAULT_MAX_TOKENS, sanitize_messages
from ypl.backend.prompts import get_system_prompt_with_modifiers
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import AssistantSelectionSource, ChatMessage, PromptModifierAssoc, MessageType
from ypl.db.language_models import LanguageModel


class StreamResponse:
    """Refer Server-sent events for details on the format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """

    def __init__(self, data: dict, event: str | None = None):
        self.data = data
        self.event = event

    def encode(self) -> str:
        response = ""
        if self.event:
            response += f"event: {self.event}\n"
        response += f"data: {json.dumps(self.data)}\n\n"
        return response


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt string")
    model: str = Field(..., description="Model internal name")
    is_new_chat: bool = True
    chat_id: uuid.UUID = Field(..., description="Chat Id  of generated on client")
    turn_id: uuid.UUID = Field(..., description="Turn Id of Chat generated on client")
    message_id: uuid.UUID = Field(..., description="Message Id generated on client")
    creator_user_id: str = Field(..., description="User Id")
    turn_seq_num: int = Field(..., description="Turn Sequence Number generated on client")
    assistant_selection_source: AssistantSelectionSource = Field(
        default=AssistantSelectionSource.UNKNOWN,
        description="Assistant selection source of model, if it were User/Router selected",
    )
    # TODO(bhanu) - make below mandatory after UI change
    prompt_modifier_ids: list[uuid.UUID] | None = Field(None, description="List of Prompt Modifier IDs")
    attachment_ids: list[uuid.UUID] | None = Field(None, description="List of Attachment IDs")
    load_existing: bool = Field(
        default=False,
        description="If true, return an existing message matching the request if it exists; otherwise create a new one",
    )
    use_all_models_in_chat_history: bool = Field(
        default=False,
        description="If true, include the chat history of all responding models.",
    )


STREAMING_ERROR_TEXT: str = "\n\\<streaming stopped unexpectedly\\>"
STOPPED_STREAMING: str = "\n\n*You stopped this response*"

router = APIRouter()


@router.post("/chat/completions")
async def chat_completions(
    chat_request: ChatRequest,
) -> StreamingResponse:
    """
    Unified streaming endpoint that supports multiple model providers

    Args:
        chat_request: The chat prompt with model selection
        background_tasks: FastAPI background tasks
    """
    try:
        client: BaseChatModel = await get_provider_client(chat_request.model)
        return StreamingResponse(_stream_chat_completions(client, chat_request), media_type="text/event-stream")
    except Exception as e:
        logging.error(f"Error initializing model: {e} \n" + traceback.format_exc())
        return StreamingResponse(
            error_stream(f"Error initializing model: {str(e)}"), media_type="text/event-stream", status_code=500
        )


async def _stream_chat_completions(client: BaseChatModel, chat_request: ChatRequest) -> AsyncIterator[str]:
    eager_persist_task: asyncio.Task[Any] | None = None
    stop_stream_task: asyncio.Task[Any] | None = None
    try:
        start_time = datetime.now()
        # Create task to eagerly persist user message
        # This is a product requirement to enable "I prefer this" button
        # before both side-by-side streams finish generating their responses.
        # The eager persistence allows users to select their preferred response
        # without waiting for complete generation.
        eager_persist_task = asyncio.create_task(
            upsert_chat_message(
                intent=Intent.EAGER_PERSIST,
                turn_id=chat_request.turn_id,
                message_id=chat_request.message_id,
                model=chat_request.model,
                message_type=MessageType.ASSISTANT_MESSAGE,
                turn_seq_num=chat_request.turn_seq_num,
                assistant_selection_source=chat_request.assistant_selection_source,
                prompt_modifier_ids=chat_request.prompt_modifier_ids,
            )
        )
        # Create task keep checking for "Stop Stream" signal from user
        stop_stream_task = asyncio.create_task(
            stop_stream_check(chat_id=chat_request.chat_id, turn_id=chat_request.turn_id, model=chat_request.model)
        )
        intial_status = {"status": "started", "timestamp": start_time.isoformat(), "model": chat_request.model}
        final_status = {"status": "completed", "model": chat_request.model}

        existing_message = None
        if chat_request.load_existing:
            existing_message = await _get_message(chat_request)

        if existing_message:
            log_dict = {
                "message": "Existing message found",
                "chat_id": str(chat_request.chat_id),
                "turn_id": str(chat_request.turn_id),
                "message_id": str(existing_message.message_id),
                "content_length": str(len(existing_message.content)),
                "model": chat_request.model,
            }
            logging.info(json_dumps(log_dict))
            # Send initial status.
            yield StreamResponse(
                intial_status | {"existing_response": True, "message_id": str(existing_message.message_id)},
                "status",
            ).encode()

            # Send the content of the existing message as a single chunk.
            yield StreamResponse({"content": existing_message.content, "model": chat_request.model}, "content").encode()

            # Send completion status.
            end_time = datetime.now()
            yield StreamResponse(
                final_status
                | {
                    "duration_ms": (end_time - start_time).total_seconds() * 1000,
                    "response_tokens": len(existing_message.content.split()),
                },
                "status",
            ).encode()
            return

        # Send initial status
        yield StreamResponse(intial_status, "status").encode()
        system_prompt = get_system_prompt_with_modifiers(
            chat_request.model, chat_request.prompt_modifier_ids, chat_request.use_all_models_in_chat_history
        )

        messages: list[BaseMessage] = []
        if system_prompt and not chat_request.model.startswith("o1"):
            # use system prompt for non o1 models. o1 doesn't support system prompt
            messages.append(SystemMessage(content=system_prompt))
        if not chat_request.is_new_chat:
            chat_history = await get_curated_chat_context(
                chat_request.chat_id, chat_request.use_all_models_in_chat_history, chat_request.model
            )
            language_model = await get_language_model(chat_request.model)
            chat_context = sanitize_messages(
                chat_history, system_prompt, language_model.context_window_tokens or DEFAULT_MAX_TOKENS
            )
            messages.extend(chat_context)
            # defensive check for race condition that user message is inserted by FE and loaded as part of history
            # so the user prompt will appear twice consecutively, which will fail for llama models.
            # This will be removed in V2
            # Check if chat_context is not empty and the last message matches the current prompt
            should_append_message = True
            last_message = chat_context[-1]
            if last_message and isinstance(last_message, HumanMessage) and last_message.content == chat_request.prompt:
                should_append_message = False
            if should_append_message:
                messages.append(HumanMessage(content=chat_request.prompt))
        else:
            messages.append(HumanMessage(content=chat_request.prompt))

        first_token_timestamp: float = 0
        response_tokens_num = 0
        full_response = ""
        message_metadata: dict[str, Any] = {}
        chunk = None
        eager_persist_task_yielded = False
        try:
            async for chunk in client.astream(messages):
                # TODO(bhanu) - assess if we should customize chunking for optimal network performance
                if hasattr(chunk, "content") and chunk.content:
                    if first_token_timestamp == 0:
                        first_token_timestamp = datetime.now().timestamp()
                    response_tokens_num += 1
                    content = chunk.content
                    # Only log in local environment
                    if settings.ENVIRONMENT == "local":
                        logging.info(chunk.content)
                    full_response += str(content)
                    yield StreamResponse({"content": content, "model": chat_request.model}).encode()
                if hasattr(chunk, "response_metadata") and chunk.response_metadata is not None:
                    if chunk.response_metadata:
                        message_metadata = add_metadata(message_metadata, chunk.response_metadata)
                        yield StreamResponse(
                            {"metadata": chunk.response_metadata, "model": chat_request.model}
                        ).encode()
                # While streaming, yield (only once) if message was eager persisted.
                if eager_persist_task.done() and not eager_persist_task_yielded:
                    eager_persist_task_yielded = True
                    yield StreamResponse(
                        {
                            "status": "message_eager_persisted",
                            "timestamp": datetime.now().isoformat(),
                            "model": chat_request.model,
                        },
                        "status",
                    ).encode()
                if stop_stream_task.done():
                    logging.info("Stop task done")
                    # Get result from stop_stream_task
                    stop_requested = await stop_stream_task
                    if stop_requested:
                        full_response += STOPPED_STREAMING
                        logging.info(
                            "Stop request received for chat "
                            f"{chat_request.chat_id}, turn {chat_request.turn_id}, model {chat_request.model}"
                        )
                        yield StreamResponse(
                            {
                                "status": "stop_stream_requested",
                                "timestamp": datetime.now().isoformat(),
                                "model": chat_request.model,
                            },
                            "status",
                        ).encode()
                        # respect the stop signal and break the stream processing
                        break
        except asyncio.CancelledError:
            logging.warning("Cancelled Error (while streaming) because of client disconnect")
        except Exception as e:
            full_response += STREAMING_ERROR_TEXT
            chunk_str = str(chunk) if chunk is not None else "No chunk available"
            logging.error(
                f"Streaming Error for model {chat_request.model} with prompt {chat_request.prompt}: {str(e)} \n"
                f"Chat ID: {chat_request.chat_id}, Message ID: {chat_request.message_id}\n"
                f"Last chunk: {chunk_str}\n"
                f"{traceback.format_exc()}"
            )
            yield StreamResponse(
                {"error": f"Streaming error: {str(e)}", "code": "stream_error", "model": chat_request.model}, "error"
            ).encode()

        # Send completion status
        end_time = datetime.now()
        yield StreamResponse(
            final_status
            | {
                "duration_ms": (end_time - start_time).total_seconds() * 1000,
                "response_tokens": response_tokens_num,
            },
            "status",
        ).encode()

        modelResponseTelemetry = ModelResponseTelemetry(
            requestTimestamp=start_time.timestamp() * 1000,  # todo - review this logic of timestamp format of FE vs BE
            firstTokenTimestamp=first_token_timestamp * 1000,
            lastTokenTimestamp=end_time.timestamp() * 1000,
            completionTokens=response_tokens_num,
        )

        try:
            # upsert
            await upsert_chat_message(
                intent=Intent.FINAL_PERSIST,
                turn_id=chat_request.turn_id,
                message_id=chat_request.message_id,
                model=chat_request.model,
                message_type=MessageType.ASSISTANT_MESSAGE,
                turn_seq_num=chat_request.turn_seq_num,
                prompt_modifier_ids=chat_request.prompt_modifier_ids,
                assistant_selection_source=chat_request.assistant_selection_source,
                content=full_response,
                streaming_metrics=modelResponseTelemetry.model_dump(),
                message_metadata=message_metadata,
            )

            # Send persistence success status
            yield StreamResponse(
                {"status": "message_persisted", "timestamp": datetime.now().isoformat(), "model": chat_request.model},
                "status",
            ).encode()
        except Exception as e:
            logging.error(
                f"Persistence error : chat_id {chat_request.chat_id}, message_id {chat_request.message_id}: {str(e)}\n"
                f"Full response: {full_response}\n"
                f"{traceback.format_exc()}"
            )
            yield StreamResponse(
                {
                    "error": "Failed to persist message",
                    "code": "persistence_error",
                    "model": str(chat_request.model),
                    "message_id": str(chat_request.message_id),
                },
                "error",
            ).encode()

    except asyncio.CancelledError as e:
        logging.warning(
            f"Cancelled Error (after streaming) because of client disconnect {str(e)} " + traceback.format_exc()
        )

    except Exception as e:
        logging.error(
            f"Error for model {chat_request.model} with prompt {chat_request.prompt}, \n"
            f"chat_id {chat_request.chat_id}, message_id {chat_request.message_id}: {str(e)} \n"
            + traceback.format_exc()
        )
        yield StreamResponse(
            {"error": f"Error: {str(e)}", "code": "error", "model": chat_request.model}, "error"
        ).encode()
    finally:
        # Cancel any pending tasks
        logging.info("Cancelling any pending tasks for " + str(chat_request.chat_id))
        if eager_persist_task is not None:
            if not eager_persist_task.done():
                eager_persist_task.cancel()
            try:
                await eager_persist_task
            except asyncio.CancelledError:
                logging.info("eager_persist_task was cancelled" + chat_request.model)

        if stop_stream_task is not None:
            if not stop_stream_task.done():
                stop_stream_task.cancel()
            try:
                await stop_stream_task
            except asyncio.CancelledError:
                logging.info("stop_stream_task was cancelled " + chat_request.model)


async def error_stream(error_message: str) -> AsyncIterator[str]:
    yield StreamResponse({"error": error_message, "code": "initialization_error"}, "error").encode()


def add_metadata(message_metadata: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    merged_metadata = message_metadata.copy()
    if "citations" in metadata and metadata["citations"]:
        merged_metadata.update({"citations": metadata["citations"]})
    return merged_metadata


async def _get_message(chat_request: ChatRequest) -> ChatMessage | None:
    """Returns a message matching all conditions in the chat request, or None if no such message is found"""
    query = (
        select(ChatMessage)
        .where(
            ChatMessage.turn_id == chat_request.turn_id,
            ChatMessage.deleted_at.is_(None),  # type: ignore
            ChatMessage.turn_sequence_number == chat_request.turn_seq_num,
            LanguageModel.internal_name == chat_request.model,
        )
        .join(LanguageModel)
        .join(PromptModifierAssoc)
        .group_by(ChatMessage.message_id)  # type: ignore
        .having(
            func.array_agg(PromptModifierAssoc.prompt_modifier_id) == chat_request.prompt_modifier_ids  # type: ignore
            if chat_request.prompt_modifier_ids
            else True,
        )
    )

    async with AsyncSession(get_async_engine()) as session:
        result = await session.execute(query)
        return result.scalar_one_or_none()


async def stop_stream_check(chat_id: uuid.UUID, turn_id: uuid.UUID, model: str) -> bool:
    while True:
        should_stop = await check_for_stop_request(chat_id, turn_id, model)
        if should_stop:
            logging.info(f"Stopping stream for chat_id={chat_id}, turn_id={turn_id}, model={model}")
            return True
        await asyncio.sleep(0.5)  # Check every 500ms
