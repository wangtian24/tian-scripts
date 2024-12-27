import json
import logging
import traceback
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from ypl.backend.config import settings
from ypl.backend.llm.chat import get_curated_chat_context, persist_chat_message
from ypl.backend.llm.model.model import ModelResponseTelemetry
from ypl.backend.llm.provider.provider_clients import get_language_model, get_provider_client
from ypl.backend.llm.sanitize_messages import DEFAULT_MAX_TOKENS, sanitize_messages
from ypl.backend.prompts import get_system_prompt_with_modifiers
from ypl.db.chats import AssistantSelectionSource


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


STREAMING_ERROR_TEXT: str = "\n\\<streaming stopped unexpectedly\\>"

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
    try:
        start_time = datetime.now()

        # Send initial status
        yield StreamResponse(
            {"status": "started", "timestamp": start_time.isoformat(), "model": chat_request.model}, "status"
        ).encode()
        system_prompt = get_system_prompt_with_modifiers(chat_request.model, chat_request.prompt_modifier_ids)

        messages: list[dict[str, str]] = []
        if system_prompt and not chat_request.model.startswith("o1"):
            # use system prompt for non o1 models. o1 doesn't support system prompt
            messages.append({"role": "system", "content": system_prompt})
        if not chat_request.is_new_chat:
            chat_history = await get_curated_chat_context(chat_request.chat_id)
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
            if (
                chat_context
                and chat_context[-1]["role"] == "user"
                and chat_context[-1]["content"] == chat_request.prompt
            ):
                should_append_message = False

            if should_append_message:
                messages.append({"role": "user", "content": chat_request.prompt})
        else:
            messages.append({"role": "user", "content": chat_request.prompt})

        first_token_timestamp: float = 0
        response_tokens_num = 0
        full_response = ""
        message_metadata = dict[str, Any]()
        chunk = None  # Define chunk outside the try block
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
            {
                "status": "completed",
                "duration_ms": (end_time - start_time).total_seconds() * 1000,
                "model": chat_request.model,
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
            await persist_chat_message(
                turn_id=chat_request.turn_id,
                message_id=chat_request.message_id,
                content=full_response,
                model=chat_request.model,
                turn_seq_num=chat_request.turn_seq_num,
                streaming_metrics=modelResponseTelemetry.model_dump(),
                prompt_modifier_ids=chat_request.prompt_modifier_ids,
                assistant_selection_source=chat_request.assistant_selection_source,
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
                    "model": chat_request.model,
                    "message_id": chat_request.message_id,
                },
                "error",
            ).encode()

    except Exception as e:
        logging.error(
            f"Error for model {chat_request.model} with prompt {chat_request.prompt}, \n"
            f"chat_id {chat_request.chat_id}, message_id {chat_request.message_id}: {str(e)} \n"
            + traceback.format_exc()
        )
        yield StreamResponse(
            {"error": f"Error: {str(e)}", "code": "error", "model": chat_request.model}, "error"
        ).encode()


async def error_stream(error_message: str) -> AsyncIterator[str]:
    yield StreamResponse({"error": error_message, "code": "initialization_error"}, "error").encode()


def add_metadata(message_metadata: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    merged_metadata = message_metadata.copy()
    if "citations" in metadata and metadata["citations"]:
        merged_metadata.update({"citations": metadata["citations"]})
    return merged_metadata
