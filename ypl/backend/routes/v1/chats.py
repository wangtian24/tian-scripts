import asyncio
import logging
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from tenacity import retry, stop_after_attempt, wait_exponential

from ypl.backend.config import settings
from ypl.backend.db import get_async_engine
from ypl.backend.jobs.tasks import store_language_code
from ypl.backend.llm.chat import ModelInfo, get_active_prompt_modifiers, get_chat_history, get_chat_model
from ypl.backend.llm.constants import ChatProvider
from ypl.backend.llm.judge import DEFAULT_PROMPT_DIFFICULTY, YuppPromptDifficultyWithCommentLabeler
from ypl.backend.llm.labeler import QT_CANT_ANSWER, MultiLLMLabeler, QuickTakeGenerator
from ypl.backend.llm.moderation import DEFAULT_MODERATION_RESULT, amoderate
from ypl.backend.llm.vendor_langchain_adapter import GeminiLangChainAdapter, OpenAILangChainAdapter
from ypl.backend.rw_cache import TurnQualityCache
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import Chat, ChatMessage, MessageType, TurnQuality
from ypl.utils import tiktoken_trim

router = APIRouter()
llm = get_chat_model(
    ModelInfo(
        provider=ChatProvider.OPENAI,
        model="gpt-4o",
        api_key=settings.OPENAI_API_KEY,
    ),
    temperature=0.0,
)

gpt_4o_mini_llm = OpenAILangChainAdapter(
    model_info=ModelInfo(
        provider=ChatProvider.OPENAI,
        model="gpt-4o-mini",
        api_key=settings.OPENAI_API_KEY,
    ),
    model_config_=dict(
        temperature=0.0,
        max_tokens=40,
    ),
)

openai_llm = OpenAILangChainAdapter(
    model_info=ModelInfo(
        provider=ChatProvider.OPENAI,
        model="gpt-4o",
        api_key=settings.OPENAI_API_KEY,
    ),
    model_config_=dict(
        temperature=0.0,
        max_tokens=40,
    ),
)

gemini_15_flash_llm = GeminiLangChainAdapter(
    model_info=ModelInfo(
        provider=ChatProvider.GOOGLE,
        model="gemini-1.5-flash-002",
        api_key=settings.GOOGLE_API_KEY,
    ),
    model_config_=dict(
        project_id=settings.GCP_PROJECT_ID,
        region=settings.GCP_REGION,
        temperature=0.0,
        max_output_tokens=40,
        top_k=1,
    ),
)


gemini_2_flash_llm = GeminiLangChainAdapter(
    model_info=ModelInfo(
        provider=ChatProvider.GOOGLE,
        model="gemini-2.0-flash-exp",
        api_key=settings.GOOGLE_API_KEY,
    ),
    model_config_=dict(
        project_id=settings.GCP_PROJECT_ID,
        region=settings.GCP_REGION_GEMINI_2,
        temperature=0.0,
        max_output_tokens=40,
        top_k=1,
    ),
)

QT_LLMS = {
    "gpt-4o": openai_llm,
    "gpt-4o-mini": gpt_4o_mini_llm,
    "gemini-1.5-flash": gemini_15_flash_llm,
    "gemini-2.0-flash": gemini_2_flash_llm,
}

QT_MAX_CONTEXT_LENGTH = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 16000,
}
# Models to use if no specific model was requested.
MODELS_FOR_DEFAULT_QT = ["gpt-4o", "gpt-4o-mini"]
# Model to use while supplying only the prompts from the chat history, instead of the full chat history.
MODEL_FOR_PROMPT_ONLY = "gpt-4o"
# Maximum number of pinned chats for a user.
MAX_PINNED_CHATS = 10


class QuickTakeResponse(BaseModel):
    quicktake: str
    model: str


class QuickTakeRequest(BaseModel):
    prompt: str | None = None
    model: str | None = None  # one of the entries in QT_LLMS; if none, use MODELS_FOR_DEFAULT_QT
    timeout_secs: float = settings.DEFAULT_QT_TIMEOUT_SECS


class PromptModifierInfo(BaseModel):
    prompt_modifier_id: str
    name: str
    description: str | None = None


def get_quicktake_generator(
    model: str,
    chat_history: list[dict[str, Any]],
    prompt_only: bool = False,
    timeout_secs: float = settings.DEFAULT_QT_TIMEOUT_SECS,
) -> QuickTakeGenerator:
    """Get a quicktake generator for a given model, or raise if the model is not supported."""
    if prompt_only:
        # Use only the prompts from the chat history.
        chat_history = [m for m in chat_history if m["role"] == "user"]
    return QuickTakeGenerator(QT_LLMS[model], chat_history, timeout_secs=timeout_secs)


async def generate_quicktake(
    request: QuickTakeRequest, chat_id: str | None = None, turn_id: str | None = None
) -> QuickTakeResponse:
    chat_history = [] if chat_id is None else get_chat_history(chat_id, turn_id)
    response_model = ""
    timeout_secs = request.timeout_secs
    try:
        if not request.model:
            # Default: use multiple models
            labelers: dict[str, Any] = {
                model: get_quicktake_generator(model, chat_history, timeout_secs=timeout_secs)
                for model in MODELS_FOR_DEFAULT_QT
            }
            # Add a fast model that uses the prompts only in the chat history.
            labelers[MODEL_FOR_PROMPT_ONLY + ":prompt-only"] = get_quicktake_generator(
                MODEL_FOR_PROMPT_ONLY, chat_history, prompt_only=True, timeout_secs=timeout_secs
            )
            multi_generator = MultiLLMLabeler(
                labelers=labelers,
                timeout_secs=timeout_secs,
                early_terminate_on=MODELS_FOR_DEFAULT_QT,
            )
            max_context_length = min((QT_MAX_CONTEXT_LENGTH[model] for model in MODELS_FOR_DEFAULT_QT), default=16000)
            quicktakes = await multi_generator.alabel(
                tiktoken_trim(request.prompt or "", int(max_context_length * 0.75), direction="right")
            )
            quicktake = QT_CANT_ANSWER
            for model in labelers:
                response = quicktakes.get(model)
                if response and not isinstance(response, Exception):
                    response_model = model
                    quicktake = response
                    break
        elif request.model in QT_LLMS:
            # Specific model requested.
            generator = get_quicktake_generator(request.model, chat_history)
            max_context_length = QT_MAX_CONTEXT_LENGTH.get(request.model, min(QT_MAX_CONTEXT_LENGTH.values()))
            quicktake = await generator.alabel(
                tiktoken_trim(request.prompt or "", int(max_context_length * 0.75), direction="right")
            )
            response_model = request.model
        else:
            raise ValueError(f"Unsupported model: {request.model}; supported: {','.join(QT_LLMS.keys())}")
    except Exception as e:
        log_dict = {
            "message": "Error generating quicktake",
            "model": request.model,
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=400, detail="Error generating quicktake") from e

    return QuickTakeResponse(quicktake=quicktake, model=response_model)


@router.post("/chats/{chat_id}/generate_quicktake", response_model=QuickTakeResponse)
async def generate_quicktake_chat_id(
    request: QuickTakeRequest,
    chat_id: str = Path(..., description="The ID of the chat"),
) -> QuickTakeResponse:
    return await generate_quicktake(request, chat_id)


@router.post("/chats/{chat_id}/turns/{turn_id}/generate_quicktake", response_model=QuickTakeResponse)
async def generate_quicktake_turn_id(
    request: QuickTakeRequest,
    chat_id: str = Path(..., description="The ID of the chat"),
    turn_id: str = Path(..., description="The ID of the turn"),
) -> QuickTakeResponse:
    return await generate_quicktake(request, chat_id, turn_id)


@router.post("/chats/{chat_id}/turns/{turn_id}:label_quality", response_model=TurnQuality)
async def label_quality(chat_id: UUID, turn_id: UUID) -> TurnQuality:
    cache = TurnQualityCache.get_instance()
    tq = await cache.aread(turn_id, deep=True)

    if not tq:
        async with AsyncSession(get_async_engine()) as session:
            tq = TurnQuality(
                turn_id=turn_id,
                chat_id=chat_id,
            )

            session.add(tq)
            await session.commit()
            await session.refresh(tq)
            session.expunge(tq)
    else:
        # Only return if we have both moderation and difficulty results.
        if tq.prompt_difficulty is not None and tq.prompt_is_safe is not None:
            return tq

    turn = tq.turn

    if turn.chat_id != chat_id:
        raise HTTPException(status_code=404, detail="Turn quality not found")

    prompt = next((m.content for m in turn.chat_messages if m.message_type == MessageType.USER_MESSAGE), None)
    responses = [m.content for m in turn.chat_messages if m.message_type == MessageType.ASSISTANT_MESSAGE]

    if prompt is None:
        raise HTTPException(status_code=400, detail="Not enough prompts or responses to label quality")

    responses += ["", ""]  # ensure at least two responses

    tasks: list[tuple[str, Any]] = []
    if tq.prompt_difficulty is None:
        labeler = YuppPromptDifficultyWithCommentLabeler(llm)
        tasks.append(("difficulty", labeler.alabel_full((prompt,) + tuple(responses[:2]))))

    if tq.prompt_is_safe is None:
        tasks.append(("moderate", amoderate(prompt)))

    if tasks:
        results = await asyncio.gather(*(task[1] for task in tasks), return_exceptions=True)

        for (task_type, _), result in zip(tasks, results, strict=True):
            if isinstance(result, Exception):
                log_dict = {
                    "message": f"Error in {task_type} task; using default value",
                    "turn_id": str(turn_id),
                    "error": str(result),
                }
                logging.warning(json_dumps(log_dict))
                if task_type == "difficulty":
                    prompt_difficulty = DEFAULT_PROMPT_DIFFICULTY
                elif task_type == "moderate":
                    moderation_result = DEFAULT_MODERATION_RESULT
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
            else:
                if task_type == "difficulty":
                    prompt_difficulty, prompt_difficulty_details = result  # type: ignore
                    tq.prompt_difficulty = prompt_difficulty
                    tq.prompt_difficulty_details = prompt_difficulty_details
                elif task_type == "moderate":
                    moderation_result = result  # type: ignore
                    tq.prompt_is_safe = moderation_result.safe
                    tq.prompt_moderation_model_name = moderation_result.model_name
                    if not moderation_result.safe:
                        tq.prompt_unsafe_reasons = moderation_result.reasons
                else:
                    raise ValueError(f"Unknown task type: {task_type}")

    try:
        cache.write(key=turn_id, value=tq)
    except Exception as e:
        log_dict = {
            "message": "Error writing turn quality to cache",
            "turn_id": str(turn_id),
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=f"Error writing turn quality to cache: {e}") from e

    for message in turn.chat_messages:
        store_language_code.delay(message.message_id, message.content)

    return tq


@router.get("/chats/{chat_id}/turns/{turn_id}/quality", response_model=TurnQuality)
async def get_quality(chat_id: UUID, turn_id: UUID) -> TurnQuality:
    cache = TurnQualityCache.get_instance()
    tq = await cache.aread(turn_id)

    if tq is None:
        raise HTTPException(status_code=404, detail="Turn quality not found")

    return tq


class PinChatRequest(BaseModel):
    pin: bool
    user_id: str


@router.patch("/chats/{chat_id}/pin", response_model=Chat)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
async def pin_chat(
    chat_id: UUID,
    pin_request: PinChatRequest,
) -> Chat:
    """
    Pin or unpin a chat for a user.
    """
    async with AsyncSession(get_async_engine()) as session:
        async with session.begin():  # Use a transaction
            # Get the chat and verify ownership
            try:
                stmt = select(Chat).where(Chat.chat_id == chat_id)  # type: ignore[arg-type]
                result = await session.execute(stmt)
                chat = result.scalar_one_or_none()
                if not chat:
                    raise HTTPException(status_code=404, detail="Chat not found")
                if str(chat.creator_user_id) != pin_request.user_id:
                    raise HTTPException(status_code=403, detail="Not authorized to modify this chat")

                if pin_request.pin:
                    # Check current number of pinned chats
                    query = (
                        select(func.count())
                        .select_from(Chat)
                        .where(Chat.creator_user_id == pin_request.user_id)  # type: ignore[arg-type]
                        .where(Chat.deleted_at.is_(None))  # type: ignore
                        .where(Chat.is_pinned.is_(True))  # type: ignore
                    )
                    result = await session.execute(query)
                    pinned_count = result.scalar()

                    if pinned_count >= MAX_PINNED_CHATS:  # type: ignore
                        raise HTTPException(status_code=400, detail=f"Cannot pin more than {MAX_PINNED_CHATS} chats")

                # Update pin status
                chat.is_pinned = pin_request.pin
                await session.flush()

                # Create a detached copy before closing the session
                chat_dict = {
                    "chat_id": chat.chat_id,
                    "creator_user_id": chat.creator_user_id,
                    "created_at": chat.created_at,
                    "modified_at": chat.modified_at,
                    "deleted_at": chat.deleted_at,
                    "is_pinned": chat.is_pinned,
                    "title": chat.title,
                    "path": chat.path,
                    "is_public": chat.is_public,
                }

                return Chat(**chat_dict)

            except HTTPException as he:
                raise HTTPException(status_code=he.status_code, detail=he.detail) from he
            except Exception as e:
                log_dict = {
                    "message": "Failed to pin/unpin chat",
                    "chat_id": str(chat_id),
                    "user_id": pin_request.user_id,
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                raise HTTPException(status_code=500, detail="Failed to pin/unpin chat") from e


@router.get("/chats/pinned", response_model=list[Chat])
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
async def get_user_pinned_chats(
    user_id: Annotated[str, Query(title="The ID of the user", required=True)],
) -> list[Chat]:
    """
    Get pinned chats for a user.

    Args:
        user_id (UUID): The UUID of the user

    Returns:
        list[Chat]: List of pinned Chat objects, ordered by creation date descending
    """
    try:
        async with AsyncSession(get_async_engine()) as session:
            query = (
                select(Chat)
                .where(Chat.creator_user_id == user_id)  # type: ignore[arg-type]
                .where(Chat.deleted_at.is_(None))  # type: ignore
                .where(Chat.is_pinned.is_(True))  # type: ignore
                .order_by(Chat.created_at.desc())  # type: ignore
                .limit(MAX_PINNED_CHATS)
            )

            result = await session.execute(query)
            chats = result.scalars().all()
            session.expunge_all()
            return list(chats)

    except Exception as e:
        log_dict = {
            "message": "Failed to fetch pinned chats",
            "user_id": str(user_id),
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail="Failed to fetch pinned chats") from e


class ChatResponse(BaseModel):
    """Response model for chat listing."""

    chats: list[Chat]
    has_more: bool


class GetChatsParams(BaseModel):
    """Parameters for fetching chat history."""

    user_id: Annotated[str, Query(title="The ID of the user", required=True)]
    last_chat_time: Annotated[
        datetime | None,
        Query(
            title="Timestamp of the last chat seen",
            description="Must be provided together with last_chat_id",
            default=None,
        ),
    ]
    last_chat_id: Annotated[
        UUID | None,
        Query(
            title="ID of the last chat seen", description="Must be provided together with last_chat_time", default=None
        ),
    ]
    page_size: Annotated[
        int,
        Query(title="Number of records to fetch", description="Number of chats to return per page", default=50, ge=1),
    ]

    @validator("user_id")
    def validate_user_id(cls, v: str) -> str:
        """Ensure user_id is not empty."""
        if not v or not v.strip():
            raise ValueError("user_id is required and cannot be empty")
        return v.strip()

    @validator("last_chat_id")
    def validate_last_chat_pair(cls, v: UUID | None, values: dict) -> UUID | None:
        """Ensure both chat values are provided if one is provided."""
        has_time = values.get("last_chat_time") is not None
        has_id = v is not None

        if has_time != has_id:
            raise ValueError(
                "Invalid pagination parameters: "
                "last_chat_time and last_chat_id must either both be provided or both be None. "
                f"Received: last_chat_time={'present' if has_time else 'missing'}, "
                f"last_chat_id={'present' if has_id else 'missing'}"
            )
        return v


@router.get("/chats", response_model=ChatResponse)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
async def get_user_chats(params: Annotated[GetChatsParams, Depends()]) -> ChatResponse:
    """
    Get non-deleted chats for a user with pagination for infinite scroll.

    If no last chat time or id is provided, returns the most recent chats.

    Args:
        params: Validated parameters for fetching chats.

    Returns:
        ChatResponse: List of chats and pagination info.

    Raises:
        HTTPException: If there's an error fetching chats.
    """
    try:
        async with AsyncSession(get_async_engine()) as session:
            query = (
                select(Chat)
                .where(Chat.creator_user_id == params.user_id)  # type: ignore[arg-type]
                .where(Chat.deleted_at.is_(None))  # type: ignore
            )

            # Only apply filtering if both parameters are provided
            if params.last_chat_time and params.last_chat_id:
                query = query.where(
                    (Chat.created_at, Chat.chat_id) < (params.last_chat_time, params.last_chat_id)  # type: ignore
                )

            query = query.order_by(Chat.created_at.desc(), Chat.chat_id.desc()).limit(params.page_size + 1)  # type: ignore

            result = await session.execute(query)
            chats = result.scalars().all()
            session.expunge_all()

            has_more = len(chats) > params.page_size
            result_chats = list(chats[: params.page_size])

            return ChatResponse(chats=result_chats, has_more=has_more)

    except Exception as e:
        log_dict = {
            "message": "Failed to fetch chat history",
            "params": params.dict(),
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail={"message": "Failed to fetch chat history", "error": str(e)}) from e


class MessageDebugInfo(BaseModel):
    message_id: UUID
    language_code: str
    modifiers: list[tuple[str, str]]


@router.get("/chat_messages/{message_id}/debug_info", response_model=MessageDebugInfo | None)
async def get_message_debug_info(message_id: UUID) -> MessageDebugInfo | None:
    try:
        async with AsyncSession(get_async_engine()) as session:
            stmt = (
                select(ChatMessage)
                .options(selectinload(ChatMessage.prompt_modifiers))  # type: ignore
                .where(ChatMessage.message_id == message_id)  # type: ignore
            )
            result = await session.execute(stmt)
            message = result.scalar_one_or_none()

            if not message:
                return None

            modifiers = [(mod.name, mod.text) for mod in message.prompt_modifiers]
            language_code = str(message.language_code) if message.language_code else ""

            return MessageDebugInfo(
                message_id=message_id,
                language_code=language_code,
                modifiers=modifiers,
            )

    except Exception as e:
        log_dict = {
            "message": f"Error getting message debug info: {str(e)}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/chats/prompt_modifiers", response_model=list[PromptModifierInfo])
async def get_prompt_modifiers() -> list[PromptModifierInfo]:
    try:
        return [
            PromptModifierInfo(prompt_modifier_id=str(m.prompt_modifier_id), name=m.name, description=m.description)
            for m in await get_active_prompt_modifiers()
        ]
    except Exception as e:
        log_dict = {"message": f"Error getting prompt modifiers: {str(e)}"}
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
