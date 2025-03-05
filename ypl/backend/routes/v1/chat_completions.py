import asyncio
import json
import logging
import traceback
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy import func, update
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.attachments.gen_image import ImageGenCallback
from ypl.backend.config import settings
from ypl.backend.db import get_async_engine, get_async_session
from ypl.backend.jobs.tasks import astore_language_code
from ypl.backend.llm.attachment import get_attachments, link_attachments
from ypl.backend.llm.chat import (
    Intent,
    SelectIntent,
    check_for_stop_request,
    get_curated_chat_context,
    update_failed_message_status,
    upsert_chat_message,
)
from ypl.backend.llm.chat_instrumentation_service import (
    ChatInstrumentationRequest,
    EventSource,
    create_or_update_instrumentation,
)
from ypl.backend.llm.chat_title import maybe_set_chat_title
from ypl.backend.llm.crawl import enhance_citations
from ypl.backend.llm.db_helpers import is_image_generation_model
from ypl.backend.llm.embedding import embed_and_store_chat_message_embeddings
from ypl.backend.llm.memories import maybe_extract_memories
from ypl.backend.llm.model.management_common import ModelErrorType, contains_error_keywords
from ypl.backend.llm.model.model import ModelResponseTelemetry
from ypl.backend.llm.model_heuristics import ModelHeuristics
from ypl.backend.llm.prompt_suggestions import maybe_add_suggested_followups
from ypl.backend.llm.provider.provider_clients import get_language_model, get_provider_client
from ypl.backend.llm.sanitize_messages import DEFAULT_MAX_TOKENS, sanitize_messages
from ypl.backend.llm.transform_messages import TransformOptions, transform_user_messages
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.prompts import get_system_prompt_with_modifiers, talk_to_other_models_system_prompt
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.monitoring import metric_inc
from ypl.backend.utils.utils import StopWatch
from ypl.db.attachments import Attachment
from ypl.db.chats import (
    AssistantSelectionSource,
    ChatMessage,
    CompletionStatus,
    MessageModifierStatus,
    MessageType,
    PromptModifierAssoc,
    Turn,
)
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum

CITATION_EXTRACTION_TIMEOUT = 20.0
MODEL_ERROR_CACHE: TTLCache = TTLCache(maxsize=100, ttl=7200)  # 2 hours

THINKING_TAG_START = "\n\n<think>\n\n"
THINKING_TAG_END = "\n\n</think>\n\n"
TTFT_timeout = 30  # seconds
TTFT_timeout_for_reasoning = 60  # seconds


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
    prompt: str = Field(..., min_length=0, description="Prompt string")
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
        default=True,
        description="If true, return an existing message matching the request if it exists; otherwise create a new one",
    )
    use_all_models_in_chat_history: bool = Field(
        default=False,
        description="If true, include the chat history of all responding models.",
    )
    user_message_id: uuid.UUID | None = Field(None, description="Message ID of the user message")
    intent: SelectIntent | None = Field(None, description="Intent of the message")
    retry_message_id: uuid.UUID | None = Field(None, description="ID of failed message, for which we are retrying")


STREAMING_ERROR_TEXT: str = "\n\\<streaming stopped unexpectedly\\>"
STOPPED_STREAMING: str = "\n\n*You stopped this response*"

router = APIRouter()

model_heuristics = ModelHeuristics(tokenizer_type="tiktoken")


async def _message_completed(chat_request: ChatRequest, message_id: uuid.UUID, full_response: str) -> None:
    """Called when a message is completed successfully."""
    asyncio.create_task(maybe_add_suggested_followups(chat_request.chat_id, chat_request.turn_id))
    if settings.EXTRACT_MEMORIES_FROM_MESSAGES:
        asyncio.create_task(
            maybe_extract_memories(chat_request.chat_id, chat_request.turn_id, chat_request.creator_user_id)
        )
    # Wait a bit before the title update to allow cache hits on the chat history.
    asyncio.create_task(maybe_set_chat_title(chat_request.chat_id, chat_request.turn_id, sleep_secs=1.5))
    asyncio.create_task(astore_language_code(str(chat_request.message_id), full_response, sleep_secs=1.0))
    if settings.EMBED_MESSAGES_UPON_COMPLETION:
        asyncio.create_task(embed_and_store_chat_message_embeddings(message_id, full_response))


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
    _log_request_details(chat_request, message="start")
    if (
        len(chat_request.prompt.strip()) == 0
        and not chat_request.attachment_ids
        and not chat_request.intent == SelectIntent.TALK_TO_OTHER_MODELS
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User prompt must have either text or attachment",
        )
    try:
        client: BaseChatModel = await get_provider_client(chat_request.model)
        return StreamingResponse(_stream_chat_completions(client, chat_request), media_type="text/event-stream")
    except Exception as e:
        logging.error(f"Error initializing model: {e} \n" + traceback.format_exc())
        return StreamingResponse(
            error_stream(f"Error initializing model: {str(e)}"), media_type="text/event-stream", status_code=500
        )


def _log_request_details(chat_request: ChatRequest, message: str, additional_fields: dict | None = None) -> None:
    """Log details about the chat request.

    Args:
        chat_request: The chat request containing details to log
        additional_fields: Optional additional fields to include in the log
        message: The phase of the request ("start" or "end")
    """
    log_dict = {
        "message": f"chat_completions: {message} - (model: {chat_request.model})",
        "chat_id": str(chat_request.chat_id),
        "turn_id": str(chat_request.turn_id),
        "message_id": str(chat_request.message_id),
        "model": chat_request.model,
    }
    if additional_fields:
        log_dict.update(additional_fields)
    logging.info(json_dumps(log_dict))


async def _handle_existing_message(
    chat_request: ChatRequest,
    existing_message: ChatMessage,
    initial_status: dict,
    final_status: dict,
    start_time: datetime,
) -> AsyncIterator[str]:
    metric_inc("stream/chat_completions/existing_message_found")
    chat_request.message_id = existing_message.message_id
    asyncio.create_task(update_modifier_status(chat_request))
    log_dict = {
        "message": "chat_completions: Existing message found",
        "chat_id": str(chat_request.chat_id),
        "turn_id": str(chat_request.turn_id),
        "message_id": str(existing_message.message_id),
        "content_length": str(len(existing_message.content)),
        "model": chat_request.model,
    }
    if chat_request.prompt_modifier_ids:
        log_dict["prompt_modifier_ids"] = ",".join(str(id) for id in chat_request.prompt_modifier_ids)
    logging.info(json_dumps(log_dict))
    # Send initial status.
    yield StreamResponse(
        initial_status | {"existing_response": True, "message_id": str(existing_message.message_id)},
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


async def get_prev_turn_id(chat_id: uuid.UUID, turn_id: uuid.UUID) -> uuid.UUID | None:
    async with get_async_session() as session:
        query = (
            select(Turn.turn_id)
            .where(
                Turn.chat_id == chat_id,
                Turn.created_at < select(Turn.created_at).where(Turn.turn_id == turn_id),  # type: ignore
            )
            .order_by(Turn.created_at.desc())  # type: ignore
            .limit(1)
        )
        result = await session.exec(query)
        return result.first()


async def _stream_chat_completions(client: BaseChatModel, chat_request: ChatRequest) -> AsyncIterator[str]:
    eager_persist_task: asyncio.Task[Any] | None = None
    stop_stream_task: asyncio.Task[Any] | None = None
    full_response = ""
    response_tokens_num = 0
    chunks_count = 0
    stopwatch = StopWatch("stream/chat_completions/", auto_export=True)
    stopwatch.start_lap("total_time_to_first_token")
    stopwatch.start_lap("total_time_to_last_token")
    try:
        start_time = datetime.now()
        # Create task keep checking for "Stop Stream" signal from user
        stop_stream_task = asyncio.create_task(
            stop_stream_check(chat_id=chat_request.chat_id, turn_id=chat_request.turn_id, model=chat_request.model)
        )
        intial_status = {"status": "started", "timestamp": start_time.isoformat(), "model": chat_request.model}
        final_status = {"status": "completed", "model": chat_request.model}

        existing_message = None
        if chat_request.load_existing and chat_request.intent != SelectIntent.TALK_TO_OTHER_MODELS:
            existing_message = await _get_message(chat_request)

        if existing_message:
            async for existing_chunk in _handle_existing_message(
                chat_request, existing_message, intial_status, final_status, start_time
            ):
                yield existing_chunk
            stopwatch.end()
            return

        metric_inc("stream/chat_completions/num_stream_new")

        # Create task to eagerly persist user message
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
                modifier_status=MessageModifierStatus.SELECTED,
            )
        )
        # Mark other messages from the same model in the same turn as HIDDEN.
        if chat_request.intent != SelectIntent.TALK_TO_OTHER_MODELS:
            asyncio.create_task(update_modifier_status(chat_request))

        # Send initial status
        yield StreamResponse(intial_status, "status").encode()
        if chat_request.intent == SelectIntent.TALK_TO_OTHER_MODELS:
            prev_turn_id = await get_prev_turn_id(chat_request.chat_id, chat_request.turn_id)
            if not prev_turn_id:
                raise ValueError(f"No previous turn found for turn {chat_request.turn_id}")
            system_prompt = await talk_to_other_models_system_prompt(chat_request.model, prev_turn_id)
        else:
            system_prompt = get_system_prompt_with_modifiers(
                chat_request.model, chat_request.prompt_modifier_ids, chat_request.use_all_models_in_chat_history
            )

        messages: list[BaseMessage] = []
        should_append_message = True
        if chat_request.intent == SelectIntent.TALK_TO_OTHER_MODELS:
            messages.append(HumanMessage(content=system_prompt))
        else:
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

        language_model = await get_language_model(chat_request.model)
        if not chat_request.is_new_chat and chat_request.intent != SelectIntent.TALK_TO_OTHER_MODELS:
            chat_history = await get_curated_chat_context(
                chat_request.chat_id,
                chat_request.use_all_models_in_chat_history,
                chat_request.model,
                chat_request.turn_id,
                context_for_logging="chat_completions",
            )
            chat_context = sanitize_messages(
                chat_history.messages, system_prompt, language_model.context_window_tokens or DEFAULT_MAX_TOKENS
            )
            messages.extend(chat_context)
            # defensive check for race condition that user message is inserted by FE and loaded as part of history
            # so the user prompt will appear twice consecutively, which will fail for llama models.
            # This will be removed in V2
            # Check if chat_context is not empty and the last message matches the current prompt
            last_message = chat_context[-1] if chat_context else None
            if last_message and isinstance(last_message, HumanMessage) and last_message.content == chat_request.prompt:
                should_append_message = False
        stopwatch.record_split("preprocessing")

        latest_attachments: list[Attachment] = []

        if chat_request.attachment_ids:
            latest_attachments = await get_attachments(chat_request.attachment_ids)

        if should_append_message and chat_request.intent != SelectIntent.TALK_TO_OTHER_MODELS:
            latest_message = HumanMessage(
                content=chat_request.prompt,
                additional_kwargs={
                    "attachments": latest_attachments,
                    "message_id": chat_request.message_id,
                },
            )
            messages.append(latest_message)
        else:
            # Defensive check to handle race condition where user message is inserted by FE
            # and loaded as part of history but the attachments are not inserted yet against the user message.
            last_message = messages[-1] or None
            if last_message and isinstance(last_message, HumanMessage):
                last_message.additional_kwargs["attachments"] = latest_attachments
                messages[-1] = last_message

        transform_options = TransformOptions(
            image_type="thumbnail",
            use_signed_url=False,
        )
        messages = await transform_user_messages(messages, chat_request.model, transform_options)
        stopwatch.record_split("process_attachments")

        run_manager = (
            AsyncCallbackManagerForLLMRun(
                handlers=[ImageGenCallback(chat_request.message_id)],
                inheritable_handlers=[ImageGenCallback(chat_request.message_id)],
                run_id=chat_request.message_id,
            )
            if (await is_image_generation_model(chat_request.model))
            else None
        )

        first_token_timestamp: float = 0
        message_metadata: dict[str, Any] = {}
        chunk = None  # Define chunk outside the try block
        metadata_future = None
        default_citations = []
        eager_persist_task_yielded = False
        stream_completion_status: CompletionStatus = CompletionStatus.SUCCESS
        ttft_recorded = False
        claude_thinking_started = False  # Used to insert <think> tag around Claude thinking text.
        try:

            async def stream_with_timeout_on_first(async_stream: AsyncIterator, model: LanguageModel):  # type: ignore[no-untyped-def]
                first_token_timeout = await get_TTFT_timeout_for_model(model)
                # apply timeout to the first token
                yield await asyncio.wait_for(async_stream.__anext__(), first_token_timeout)
                async for chunk in async_stream:
                    yield chunk

            model_response_stream = client.astream(messages, run_manager=run_manager)

            async for chunk in stream_with_timeout_on_first(model_response_stream, language_model):
                # TODO(bhanu) - assess if we should customize chunking for optimal network performance
                if hasattr(chunk, "content") and chunk.content:
                    if first_token_timestamp == 0:
                        first_token_timestamp = datetime.now().timestamp()
                    chunks_count += 1
                    if isinstance(chunk.content, list) and chat_request.model.startswith("claude"):
                        # Convert Claude thinking chunks to flat text with <think> and </think> tags to
                        # match format of DeepSeek-R1 thinking reponse.
                        thinking_content, text_content = _process_claude_thinking_content(chunk.content)
                        content = ""
                        if thinking_content:
                            if not claude_thinking_started:
                                claude_thinking_started = True
                                content += THINKING_TAG_START
                            content += thinking_content
                        if text_content:
                            if claude_thinking_started:
                                # Assume thinking ends with first text chunk.
                                content += THINKING_TAG_END
                                claude_thinking_started = False
                            content += text_content
                    else:
                        content = str(chunk.content)  # Type is str | list.
                    # Only log in local environment
                    if settings.ENVIRONMENT == "local":
                        print(f"[{chunk.content}]")
                    full_response += str(content)
                    if not ttft_recorded:
                        stopwatch.end_lap("total_time_to_first_token")
                        ttft_recorded = True
                    yield StreamResponse({"content": content, "model": chat_request.model}).encode()
                if hasattr(chunk, "response_metadata") and chunk.response_metadata is not None:
                    if chunk.response_metadata:
                        if "citations" in chunk.response_metadata and chunk.response_metadata["citations"]:
                            default_citations = [
                                {"title": url, "description": "", "url": url}
                                for url in chunk.response_metadata["citations"]
                            ]
                            metadata_future = asyncio.create_task(
                                enhance_citations(chunk.response_metadata["citations"])
                            )
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
                        stream_completion_status = CompletionStatus.USER_ABORTED
                        # respect the stop signal and break the stream processing
                        break
        except asyncio.CancelledError:
            logging.warning("Cancelled Error (while streaming) because of client disconnect")
            stream_completion_status = CompletionStatus.SYSTEM_ERROR
        except TimeoutError as timeout_error:
            stream_completion_status = CompletionStatus.STREAMING_ERROR_FIRST_TOKEN_TIMEOUT
            full_response += STREAMING_ERROR_TEXT  # just to trigger fallback
            yield StreamResponse({"content": STREAMING_ERROR_TEXT, "model": chat_request.model}).encode()
            yield StreamResponse(
                {"error": "Timeout waiting for model response", "code": "timeout_error", "model": chat_request.model},
                "error",
            ).encode()
            logging.warning(
                json_dumps(
                    {
                        "message": "Streaming Timeout Error - First Token Timeout",
                        "model": chat_request.model,
                        "message_id": str(chat_request.message_id),
                        "turn_id": str(chat_request.turn_id),
                        "error_message": str(timeout_error),
                        "chat_id": str(chat_request.chat_id),
                        "traceback": traceback.format_exc(),
                    }
                )
            )
        except Exception as e:
            stream_completion_status = CompletionStatus.STREAMING_ERROR
            full_response += STREAMING_ERROR_TEXT
            chunk_str = str(chunk) if chunk is not None else "No chunk available"

            logging.error(
                json_dumps(
                    {
                        "message": f"Streaming Error from model [{chat_request.model}]: {str(e)[:100]}",
                        "model": chat_request.model,
                        "message_id": str(chat_request.message_id),
                        "error_message": str(e),
                        "turn_id": str(chat_request.turn_id),
                        "chat_id": str(chat_request.chat_id),
                        "last_chunk": chunk_str,
                        "traceback": traceback.format_exc(),
                    }
                )
            )

            yield StreamResponse({"content": STREAMING_ERROR_TEXT, "model": chat_request.model}).encode()
            yield StreamResponse(
                {"error": f"Streaming error: {str(e)}", "code": "stream_error", "model": chat_request.model}, "error"
            ).encode()
            log_attachments_in_conversation(messages, chat_request.message_id, chat_request.chat_id)
            # check if it's a potential billing error & async post to Slack
            error_message = str(e)
            error_type, excerpt = contains_error_keywords(error_message)
            if error_type and not recently_posted_error(chat_request.model, error_type=error_type):
                clean_excerpt = excerpt.replace("`", "\\`").replace("'", "\\'") if excerpt else ""
                asyncio.create_task(
                    post_to_slack(
                        f"*Potential {error_type.value} error*: {chat_request.model}: `... {clean_excerpt} ...`\n"
                        f"Chat ID: {chat_request.chat_id}\n"
                        f"Client: {str(client)}\n"
                    )
                )
        stopwatch.record_split("stream_message_chunks")

        # if full_response is empty, send an error message to enable retry
        if not full_response:
            logging.warning(
                json_dumps(
                    {
                        "message": "Empty response from Model",
                        "model": chat_request.model,
                        "message_id": str(chat_request.message_id),
                    }
                )
            )
            stream_completion_status = CompletionStatus.STREAMING_ERROR
            yield StreamResponse({"content": STREAMING_ERROR_TEXT, "model": chat_request.model}).encode()

        # once streaming is done, update the status of the failed message
        if chat_request.intent == SelectIntent.RETRY and chat_request.retry_message_id:
            asyncio.create_task(update_failed_message_status(chat_request.retry_message_id))

        # Send completion status
        end_time = datetime.now()
        tokenizer = model_heuristics.get_tokenizer_counter()
        response_tokens_num = tokenizer(full_response)
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
            chunks_count=chunks_count,
        )
        stopwatch.end_lap("total_time_to_last_token")
        stopwatch.record_split("postprocessing")

        asyncio.create_task(
            create_or_update_instrumentation(
                ChatInstrumentationRequest(
                    message_id=chat_request.message_id,
                    event_source=EventSource.MIND_SERVER,
                    streaming_metrics=modelResponseTelemetry.model_dump(),
                )
            )
        )

        try:
            if metadata_future:
                try:
                    citations = await asyncio.wait_for(metadata_future, timeout=CITATION_EXTRACTION_TIMEOUT)
                    message_metadata["citations"] = citations
                except TimeoutError:
                    logging.warning("Timeout error while enhancing citations")
                    message_metadata["citations"] = default_citations
                stopwatch.record_split("wait_for_citations")
                yield StreamResponse({"metadata": message_metadata, "model": chat_request.model}).encode()

            # upsert
            message_id = await upsert_chat_message(
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
                completion_status=stream_completion_status,
                modifier_status=MessageModifierStatus.SELECTED,
            )
            if chat_request.attachment_ids and chat_request.user_message_id:
                # TODO(Arun)
                # Check to see if we can schedule a task to link attachments at the begining
                # and check status later.
                await link_attachments(chat_request.user_message_id, chat_request.attachment_ids)
            if stream_completion_status == CompletionStatus.SUCCESS:
                await _message_completed(chat_request, message_id, full_response)
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
        logging.info(
            json_dumps(
                {
                    "message": "Streaming Stats",
                    "model": chat_request.model,
                    "message_id": str(chat_request.message_id),
                    "response_length": len(full_response),
                    "response_tokens": response_tokens_num,
                    "chunks_count": chunks_count,
                }
            )
        )
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

        _log_request_details(
            chat_request,
            message="end",
            additional_fields={"duration_ms": (end_time - start_time).total_seconds() * 1000},
        )

    stopwatch.end()


async def get_TTFT_timeout_for_model(language_model: LanguageModel) -> int:
    if language_model.is_reasoning:
        return TTFT_timeout_for_reasoning
    else:
        return TTFT_timeout


async def error_stream(error_message: str) -> AsyncIterator[str]:
    yield StreamResponse({"error": error_message, "code": "initialization_error"}, "error").encode()


async def _get_message(chat_request: ChatRequest) -> ChatMessage | None:
    """Returns a successfully completed message matching all conditions in the request, or None if no such message."""
    query = (
        select(ChatMessage)
        .where(
            ChatMessage.turn_id == chat_request.turn_id,
            ChatMessage.deleted_at.is_(None),  # type: ignore
            ChatMessage.turn_sequence_number == chat_request.turn_seq_num,
            LanguageModel.internal_name == chat_request.model,
            ChatMessage.completion_status.in_([CompletionStatus.SUCCESS, CompletionStatus.USER_ABORTED]),  # type: ignore[attr-defined]
        )
        .join(LanguageModel)
        .outerjoin(PromptModifierAssoc)
        .group_by(ChatMessage.message_id)  # type: ignore
        .having(
            func.array_agg(PromptModifierAssoc.prompt_modifier_id)  # type: ignore
            == (chat_request.prompt_modifier_ids if chat_request.prompt_modifier_ids else [None])  # type: ignore
        )
    )

    async with AsyncSession(get_async_engine()) as session:
        result = await session.execute(query)
        row = result.first()
        return row[0] if row else None


async def stop_stream_check(chat_id: uuid.UUID, turn_id: uuid.UUID, model: str) -> bool:
    while True:
        should_stop = await check_for_stop_request(chat_id, turn_id, model)
        if should_stop:
            logging.info(f"Stopping stream for chat_id={chat_id}, turn_id={turn_id}, model={model}")
            return True
        await asyncio.sleep(0.5)  # Check every 500ms


async def update_modifier_status(chat_request: ChatRequest) -> None:
    """Sets the status of the message in `chat_request` to SEEN and others in the turn to HIDDEN."""
    async with AsyncSession(get_async_engine()) as session:
        # Hide other messages in the turn that match the model
        hide_query = (
            update(ChatMessage)
            .where(
                ChatMessage.turn_id == chat_request.turn_id,  # type: ignore
                ChatMessage.message_id != chat_request.message_id,  # type: ignore
                ChatMessage.message_type == MessageType.ASSISTANT_MESSAGE,  # type: ignore
                ChatMessage.assistant_language_model_id
                == select(LanguageModel.language_model_id)
                .where(
                    LanguageModel.internal_name == chat_request.model,
                    LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
                )
                .scalar_subquery(),  # type: ignore
            )
            .values(modifier_status=MessageModifierStatus.HIDDEN)
        )
        await session.execute(hide_query)

        # Show the current message
        show_query = (
            update(ChatMessage)
            .where(
                ChatMessage.message_id == chat_request.message_id,  # type: ignore
                ChatMessage.message_type == MessageType.ASSISTANT_MESSAGE,  # type: ignore
            )
            .values(modifier_status=MessageModifierStatus.SELECTED)
        )
        await session.execute(show_query)

        await session.commit()


def log_attachments_in_conversation(messages: list[BaseMessage], message_id: uuid.UUID, chat_id: uuid.UUID) -> None:
    total_attachments_in_conversation = 0
    logs = []
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        content = message.content
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            image_url = part.get("image_url", {}).get("url", "")
            if image_url:
                logs.append(f"Attachments: Image in the conversation has size: {len(image_url)}")
                total_attachments_in_conversation += 1
    logs.append(f"Attachments: Total attachments in the conversation: {total_attachments_in_conversation}")
    log_dict = {"attachments": logs, "message_id": str(message_id), "chat_id": str(chat_id)}
    logging.info(json_dumps(log_dict))


def recently_posted_error(model: str, error_type: ModelErrorType) -> bool:
    # Returns True if a model error has been posted to Slack recently for the model.
    cache_key = f"{model}_{error_type.value}"
    if cache_key in MODEL_ERROR_CACHE:
        return True
    MODEL_ERROR_CACHE[cache_key] = True
    return False


def _process_claude_thinking_content(content: list) -> tuple[str, str]:
    """
    Process Claude thinking and text content blocks and return thinking and text contents as a tuple.
    Sample content (from https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking):
    [
        "content": [
            {
                "type": "thinking",
                "thinking": "Let me analyze this step by step...",
                "signature": "WaUjzkypQ2mUEVM36O2TxuC06KN8xyfbJwyem2dw3URve/op91XWHOEBLLqIOMfFG/UvLEczmEsUjavL...."
            },
            {
                "type": "redacted_thinking",
                "data": "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIwxtE3rAFBa8cr3qpP..."
            },
            {
                "type": "text",
                "text": "Based on my analysis..."
            }
        ]
    ]
    Returns a tuple of (thinking_content, text_content). Typically only one of them is non-empty.
    """

    thinking_content = ""
    text_content = ""

    for chunk in content:
        if not isinstance(chunk, dict):
            logging.warning(f"Claude thinking content is expected to be a dict, but it was {chunk}")
            continue

        if chunk["type"] == "thinking":
            if "thinking" in chunk:
                thinking_content += chunk["thinking"]
            if "signature" in chunk:
                # TODO: Handle thinking signature better. Claude API requires it for next turn. Just log for now.
                logging.info("Claude signature for thinking chunks is currently ignored")

        elif chunk["type"] == "redacted_thinking" and "data" in chunk:
            thinking_content += " (redacted thinking)"

        elif chunk["type"] == "text" and "text" in chunk:
            text_content += chunk["text"]

    return (thinking_content, text_content)
