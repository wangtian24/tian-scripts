import logging
import time
import uuid
from enum import Enum
from typing import Any
from uuid import UUID

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel
from rapidfuzz.distance import JaroWinkler
from sqlmodel import select

from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.llm.attachment import get_attachments
from ypl.backend.llm.context import get_curated_chat_context
from ypl.backend.llm.db_helpers import get_model_context_lengths
from ypl.backend.llm.labeler import QT_CANT_ANSWER, CantAnswerException, QuickTakeGenerator
from ypl.backend.llm.model_heuristics import ModelHeuristics
from ypl.backend.llm.provider.provider_clients import get_internal_provider_client
from ypl.backend.llm.transform_messages import TransformOptions, transform_user_messages
from ypl.backend.prompts import (
    SYSTEM_QUICKTAKE_FALLBACK_PROMPT,
    SYSTEM_QUICKTAKE_PROMPT,
    SYSTEM_RETAKE_PROMPT,
    USER_QUICKTAKE_FALLBACK_PROMPT,
    USER_QUICKTAKE_PROMPT,
    USER_RETAKE_PROMPT,
)
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.monitoring import metric_inc, metric_record
from ypl.backend.utils.utils import StopWatch
from ypl.db.chats import ChatMessage, MessageType
from ypl.db.language_models import LanguageModel
from ypl.utils import Delegator, get_text_part, replace_text_part, tiktoken_trim

# Models to use if no specific model was requested.
MODELS_FOR_DEFAULT_QT = [
    "Meta-Llama-3.3-70B-Instruct",  # from Sambanova
    "gpt-4o",
    "gpt-4o-mini",
    "gemini-2.0-flash-001",
]
# MODELS_FOR_DEFAULT_QT = ["Meta-Llama-3.3-70B-Instruct", "gpt-4o", "gpt-4o-mini", "gemini-2.0-flash-exp"]
# Model to use while supplying only the prompts from the chat history, instead of the full chat history.
MODEL_FOR_PROMPT_ONLY = "gpt-4o"
MODEL_FOR_PROMPT_ONLY_FULL_NAME = MODEL_FOR_PROMPT_ONLY + ":prompt-only"
# Fine-tuned model to use that minimizes truncations and formatting in responses
# More details at https://platform.openai.com/finetune/ftjob-VupgrOxNp0ApGhGKDgspdGjb
MODEL_FOR_FINETUNE_QT = "gpt-4o"
MODEL_FOR_FINETUNE_QT_FULL_NAME = "ft:gpt-4o-2024-08-06:yupp::AgJJZBsG"

# For fallback.
MODELS_FOR_FALLBACK = ["gemini-1.5-flash-002"]  # can add others later

# Attachment support
QT_MODEL_WITH_PDF_SUPPORT = ["gemini-2.0-flash-001"]
QT_MODEL_WITH_IMAGE_SUPPORT = ["gpt-4o", "gpt-4o-mini", "gemini-2.0-flash-001"]

MAX_QT_LLM_TOKENS = 40
DEFAULT_QT_MAX_CONTEXT_LENGTH = 128000  # gpt-4o-mini

RETAKE_SAME_ANSWER = "<SAME_ANSWER>"
RETAKE_QUICKTAKE_MIN_SIMILARITY = 0.7


class QuickTakeResponse(BaseModel):
    quicktake: str
    model: str
    errors: str | None = None


class QuickTakeIntent(str, Enum):
    INITIAL = "initial"
    RETAKE = "retake"


class QuickTakeRequest(BaseModel):
    user_id: str | None = None
    chat_id: str | None = None
    turn_id: str | None = None
    prompt: str | None = None
    attachment_ids: list[UUID] | None = None
    model: str | None = None  # one of the entries in QT_LLMS; if none, use MODELS_FOR_DEFAULT_QT
    timeout_secs: float | None = None
    intent: QuickTakeIntent = QuickTakeIntent.INITIAL

    def is_retake(self) -> bool:
        return self.intent == QuickTakeIntent.RETAKE


async def get_qt_llm(model_name: str) -> BaseChatModel:
    return await get_internal_provider_client(model_name, max_tokens=MAX_QT_LLM_TOKENS)


async def create_quicktake_generator(
    model: str,
    chat_history: list[BaseMessage],
    prompt_only: bool = False,
    timeout_secs: float = settings.DEFAULT_QT_TIMEOUT_SECS,
    user_prompt: str = USER_QUICKTAKE_PROMPT,
    system_prompt: str = SYSTEM_QUICKTAKE_PROMPT,
) -> QuickTakeGenerator:
    """Get a quicktake generator for a given model, or raise if the model is not supported."""
    if prompt_only:
        # Use only the prompts from the chat history.
        chat_history = [m for m in chat_history if isinstance(m, HumanMessage)]
    return QuickTakeGenerator(
        await get_qt_llm(model),
        chat_history,
        model_name=model,
        timeout_secs=timeout_secs,
        user_quicktake_prompt=user_prompt,
        system_quicktake_prompt=system_prompt,
        on_error="raise",
    )


async def _get_qt_model(turn_id: UUID) -> str | None:
    """Returns the quicktake model used in a given turn_id, or None if no quicktake found."""
    async with get_async_session() as session:
        stmt = (
            select(LanguageModel.internal_name)
            .join(ChatMessage, LanguageModel.language_model_id == ChatMessage.assistant_language_model_id)  # type: ignore
            .where(
                ChatMessage.turn_id == turn_id,
                ChatMessage.message_type == MessageType.QUICK_RESPONSE_MESSAGE,
            )
            .order_by(ChatMessage.created_at.desc())  # type: ignore
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


def _same_as_previous_quicktake(qt: str | None, rt: str | None) -> bool:
    if not rt or not qt:
        return False

    if "SAME_ANSWER" in rt or "CANT_ANSWER" in rt:
        # The LLM doesn't always use the canonical "<SAME_ANSWER>" string.
        return True

    return JaroWinkler.normalized_similarity(qt, rt) > RETAKE_QUICKTAKE_MIN_SIMILARITY


async def _get_turn_and_model(request: QuickTakeRequest) -> tuple[UUID | None, str | None]:
    """Checks the request and returns its turn_id and model if it is valid."""
    turn_id = uuid.UUID(request.turn_id) if request.turn_id else None
    if request.is_retake():
        if not turn_id:
            raise ValueError("turn_id is required for retake")
        # The requested model should match the model previously used for quicktake, or be empty.
        # If it is empty, we will use the model that was used most recently for quicktake.
        prev_qt_model = await _get_qt_model(turn_id)
        if request.model:
            if prev_qt_model and prev_qt_model != request.model:
                logging.warning(  # noqa: F821
                    json_dumps(
                        {
                            "message": "Regenerating quicktake with a different model from the original one",
                            "prev_model": prev_qt_model,
                            "new_model": request.model,
                        }
                    )
                )
        else:
            request.model = prev_qt_model
    return turn_id, request.model


async def generate_quicktake(
    request: QuickTakeRequest,
    chat_history: list[BaseMessage] | None = None,
) -> QuickTakeResponse:
    """
    Generates a quicktake for a given chat_id or chat_history. If chat_history is provided, it will be used instead of
    chat_id and turn_id.

    Args:
        chat_id: The chat ID to fetch history for.
        turn_id: The turn ID to fetch history for.
        chat_history: The chat history to use.
    """
    start_time = time.time()
    stopwatch = StopWatch("quicktake/latency/", auto_export=True)

    match request.chat_id, request.turn_id, chat_history:
        case None, None, None:
            raise ValueError("Either chat_id or chat_history must be provided")
        case None, None, _:
            pass
        case _, _, None:
            turn_id, requested_model = await _get_turn_and_model(request)
            chat_context = await get_curated_chat_context(
                chat_id=uuid.UUID(request.chat_id),
                use_all_models_in_chat_history=False,
                model=requested_model or "",
                current_turn_id=turn_id,
                context_for_logging="quicktake",
                max_turns=5,
            )
            chat_history = chat_context.messages

    assert chat_history is not None, "chat_history is null"

    chat_history_time = time.time() - start_time

    # Add attachments (image, etc) to the chat history, as a url or base64 encoded string.
    old_attachments = [attachment for m in chat_history for attachment in m.additional_kwargs.get("attachments", [])]
    new_attachments = await get_attachments(request.attachment_ids) if request.attachment_ids else []
    all_attachments = old_attachments + new_attachments
    stopwatch.record_split("get_attachments")

    has_attachments = len(all_attachments) > 0
    has_pdf_attachments = any(attachment.content_type == "application/pdf" for attachment in all_attachments)
    has_image_attachments = any(
        attachment.content_type is not None and attachment.content_type.startswith("image/")
        for attachment in all_attachments
    )
    parse_pdf_locally = settings.PARSE_PDF_LOCALLY_FOR_QUICKTAKE
    transform_options: TransformOptions = {
        "image_type": "thumbnail",
        "use_signed_url": False,
        "parse_pdf_locally": parse_pdf_locally,
        "max_pdf_text": settings.MAX_TEXT_TO_EXTRACT_FROM_PDF,
    }
    chat_history = await transform_user_messages(chat_history, QT_MODEL_WITH_PDF_SUPPORT[0], options=transform_options)
    stopwatch.record_split("transform_chat_history")

    # Calculate the length of input with all the information, and check against the context length allowed by the model.
    chat_history_text = "\n".join(get_text_part(m) for m in chat_history)
    chat_history_context_len = len(ModelHeuristics(tokenizer_type="tiktoken").encode_tokens(chat_history_text))
    # Add a buffer of 20% or 2000 tokens, whichever is larger, for system prompt etc.
    min_required_context_len = max(int(chat_history_context_len * 1.2), (chat_history_context_len + 2500))
    context_lengths = get_model_context_lengths()
    stopwatch.record_split("tokenize_and_get_context_lengths")

    # Choose models to use for generating quicktakes, we have a set of main models that try to answer the question,
    # and a set of fallback models that are fast but only try to provide contextual commentaries.
    # TODO(tian): no need to prefilter the models as we are trimming the message to fit their context length anyway.
    qt_models = [
        model
        for model in MODELS_FOR_DEFAULT_QT
        if context_lengths.get(model, DEFAULT_QT_MAX_CONTEXT_LENGTH) > min_required_context_len
    ]
    fallback_models = [
        model
        for model in MODELS_FOR_FALLBACK
        if context_lengths.get(model, DEFAULT_QT_MAX_CONTEXT_LENGTH) > min_required_context_len
    ]
    if request.is_retake():
        fallback_models = []
        current_turn_context = await get_curated_chat_context(
            chat_id=uuid.UUID(request.chat_id),
            model="",
            max_turns=1,
            current_turn_id=turn_id,
            include_current_turn=True,
            use_all_models_in_chat_history=False,
            return_all_current_turn_responses=True,
            context_for_logging="quicktake_retake",
        )
    _model_max_context_lengths = {k: v for k, v in context_lengths.items() if k in qt_models + fallback_models}

    def update_timeout(cur_timeout: float, condition: bool, new_timeout: float) -> float:
        return max(new_timeout, cur_timeout) if condition else cur_timeout

    timeout_secs = (
        request.timeout_secs  # the timeout provided by the client will override the defaults
        if request.timeout_secs
        else settings.DEFAULT_QT_TIMEOUT_SECS
    )

    timeout_secs = update_timeout(timeout_secs, has_attachments, settings.ATTACHMENT_QUICKTAKE_TIMEOUT_SECS)
    timeout_secs = update_timeout(timeout_secs, request.is_retake(), settings.RETAKE_QT_TIMEOUT_SECS)

    preferred_models = []
    if requested_model:
        preferred_models = [requested_model]
    elif has_pdf_attachments or has_image_attachments:
        if has_pdf_attachments and not parse_pdf_locally:
            preferred_models = QT_MODEL_WITH_PDF_SUPPORT
        elif has_image_attachments:
            preferred_models = QT_MODEL_WITH_IMAGE_SUPPORT
        if not preferred_models:
            logging.warning("No preferred models found, using all models")

    # Transform the latest message with attachments, if any.
    # Supply this to the labelers after trimming the context.
    latest_message_transform_result = await transform_user_messages(
        [
            HumanMessage(
                content=request.prompt or "",
                additional_kwargs={
                    "attachments": new_attachments,
                },
            )
        ],
        QT_MODEL_WITH_PDF_SUPPORT[0],
        options=transform_options,
    )
    _latest_message = HumanMessage(content=latest_message_transform_result[0].content)
    errors = ""

    stopwatch.record_split("transform_latest_msg")

    try:
        # -- Prepare labelers for the main quicktake call
        primary_models = []  # we only early terminate on these models
        if not preferred_models:
            primary_models.extend(qt_models)
        else:
            primary_models.extend(preferred_models)

        secondary_models = [MODEL_FOR_PROMPT_ONLY_FULL_NAME] if not has_attachments and not request.is_retake() else []
        fallback_models = fallback_models if not has_attachments else []

        system_prompt = SYSTEM_QUICKTAKE_PROMPT
        user_prompt = USER_QUICKTAKE_PROMPT
        if request.is_retake():
            system_prompt = SYSTEM_RETAKE_PROMPT
            user_prompt = USER_RETAKE_PROMPT

        # We have three tiers of QT models (labelers)
        # 1. primary - high quality but not the fastets, might have refusal as well. Can early terminate.
        # 2. secondary - faster but might not be as good.
        # 3. fallback - fastest but may not fully answer the question, just contextual commentaries.
        all_labelers: dict[str, Any] = {
            # primary labelers
            model: await create_quicktake_generator(
                model,
                chat_history,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )
            for model in primary_models
        }

        if secondary_models:
            all_labelers = all_labelers | {
                # secondary labelers
                MODEL_FOR_PROMPT_ONLY_FULL_NAME: await create_quicktake_generator(
                    MODEL_FOR_PROMPT_ONLY,
                    chat_history,
                    prompt_only=True,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                ),
            }

        if fallback_models:
            all_labelers = all_labelers | {
                # fallback labelers
                model: await create_quicktake_generator(
                    model,
                    chat_history,
                    user_prompt=USER_QUICKTAKE_FALLBACK_PROMPT,
                    system_prompt=SYSTEM_QUICKTAKE_FALLBACK_PROMPT,
                )
                for model in fallback_models
            }

        priority_groups = [primary_models, secondary_models, fallback_models]
        # TODO(Raghu): This includes secondary and fallback models even when there is preferred model
        #              or PDF attachment. Decide if we want to include these.

        stopwatch.record_split("prepare_tasks")

        # -- Make all quicktake calls in parallel
        class LabelerTask:
            model: str
            labeler: QuickTakeGenerator

            def __init__(self, model: str, labeler: QuickTakeGenerator):
                self.model = model
                self.labeler = labeler

            async def async_run(self) -> Any:
                max_context_length = min(
                    _model_max_context_lengths.get(self.model, DEFAULT_QT_MAX_CONTEXT_LENGTH),
                    DEFAULT_QT_MAX_CONTEXT_LENGTH,
                )
                prompt_args = {
                    "prompt": tiktoken_trim(request.prompt or "", int(max_context_length * 0.75), direction="right"),
                }
                if request.is_retake():
                    prompt_args["previous_quicktake_response"] = current_turn_context.current_turn_quicktake or ""
                    prompt_args["assistant_responses"] = ""
                    if current_turn_context.current_turn_responses:
                        for model, response in current_turn_context.current_turn_responses.items():
                            prompt_args[
                                "assistant_responses"
                            ] += f"Response from {model}:\n{response.content}\n\n---\n\n"
                trimmed_message = replace_text_part(
                    _latest_message,
                    self.labeler.user_quicktake_prompt.format(**prompt_args),
                )
                result = await self.labeler.alabel(trimmed_message)
                return result

        labeler_tasks = {m: LabelerTask(m, labeler) for m, labeler in all_labelers.items()}
        all_quicktakes: dict[str, Any] = await Delegator(
            labeler_tasks, timeout_secs=timeout_secs, priority_groups=priority_groups
        ).async_run()

        stopwatch.record_split("fetch_from_llms")

        # -- Post-processing
        response_quicktake = QT_CANT_ANSWER
        response_model = ""
        has_good_response = False
        has_cant_answer = False
        has_other_failure = False
        for model in all_labelers.keys():
            response = all_quicktakes.get(model)
            if response and not isinstance(response, Exception) and not has_good_response:
                response_model = model
                response_quicktake = response
                has_good_response = True
            elif isinstance(response, CantAnswerException):
                has_cant_answer = True
            else:
                has_other_failure = True
        if not has_good_response:
            errors = "no_response"

        if request.is_retake():
            if _same_as_previous_quicktake(current_turn_context.current_turn_quicktake, response_quicktake):
                response_quicktake = RETAKE_SAME_ANSWER

            logging.info(
                json_dumps(
                    {
                        "message": "Regenerated quicktake",
                        "model": requested_model,
                        "previous_quicktake": current_turn_context.current_turn_quicktake,
                        "new_quicktake": response_quicktake,
                        "chat_id": request.chat_id,
                        "turn_id": request.turn_id,
                    }
                )
            )

        metric_inc(f"quicktake/model_{response_model or 'NONE'}")
        if has_good_response:
            metric_inc("quicktake/num_good_response")
        elif has_cant_answer and has_other_failure:
            metric_inc("quicktake/num_failed_mixed")
        elif has_cant_answer:
            metric_inc("quicktake/num_failed_all_cannot_answer")
        else:
            metric_inc("quicktake/num_failed_all_timed_out")
        metric_inc("quicktake/num_total")

    except Exception as e:
        err_log_dict = {
            "message": "Error generating quicktake",
            "model": preferred_models,
            "error": str(e),
        }
        logging.exception(json_dumps(err_log_dict))
        raise e

    # The client is not aware of these private models, so return its base name; keep the full name in the log above.
    # TODO(tian): this is no longer important, remove this later.
    if response_model == MODEL_FOR_PROMPT_ONLY_FULL_NAME:
        response_model = MODEL_FOR_PROMPT_ONLY
    if response_model == MODEL_FOR_FINETUNE_QT_FULL_NAME:
        response_model = MODEL_FOR_FINETUNE_QT

    # Logging and bookkeeping
    end_time = time.time()
    metric_record("quicktake/latency_ms", int((end_time - start_time) * 1000))
    time_taken_ms = int((end_time - start_time) * 1000)
    log_dict: dict[str, Any] = {
        "message": f"Quicktake generated with {response_model} in {time_taken_ms}ms: {response_quicktake}",
        "chat_history": {
            "time_ms": int(chat_history_time * 1000),
            "num_messages": len(chat_history),
            "context_length": chat_history_context_len,
            "text_length": len(chat_history_text),
        },
        # TODO(Raghu): We treat it as refusal even when all the models timeout. Might treat it differently.
        "is_refusal": str(response_quicktake.strip() == QT_CANT_ANSWER),
        "chat_id": request.chat_id,
        "turn_id": request.turn_id,
        "model": response_model,
        "model_responses": [f"{model} -> {str(all_quicktakes[model])}" for model in all_labelers.keys()],
        "duration_secs": end_time - start_time,
        "content_length": len(response_quicktake),
        "old_attachments_ids": [attachment.attachment_id for attachment in old_attachments],
        "new_attachments_ids": [attachment.attachment_id for attachment in new_attachments],
        "attachment_mime_types": [attachment.content_type for attachment in all_attachments],
    }
    logging.info(json_dumps(log_dict))
    stopwatch.end("postprocessing")

    return QuickTakeResponse(quicktake=response_quicktake, model=response_model, errors=errors)
