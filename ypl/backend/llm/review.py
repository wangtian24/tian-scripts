"""Module for reviewing message accuracy."""

import asyncio
import enum
import logging
import time
from collections.abc import Mapping
from typing import Any, Generic, TypeVar, cast
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from sqlalchemy import select
from typing_extensions import TypedDict

from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.llm.chat import (
    get_curated_chat_context,
    get_gemini_2_flash_llm,
    get_gemini_15_flash_llm,
    get_gpt_4o_llm,
    get_gpt_4o_mini_llm,
)
from ypl.backend.llm.labeler import LLMLabeler
from ypl.backend.prompts import (
    BINARY_REVIEW_PROMPT,
    CRITIQUE_REVIEW_PROMPT,
    fill_cur_datetime,
)
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import Turn

# Type variable for review results
ReviewResultType = TypeVar("ReviewResultType", bool, str)


class ReviewType(str, enum.Enum):
    """Type of review to perform."""

    BINARY = "binary"
    CRITIQUE = "critique"


class ReviewStatus(str, enum.Enum):
    """Status of the review operation."""

    SUCCESS = "success"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class ReviewRequest(BaseModel):
    user_id: str | None = None
    turn_id: str | None = None
    review_types: list[ReviewType] | None = None
    prompt: str | None = None
    reviewer_model: str | None = None
    timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS


class BinaryResult(TypedDict):
    """Result from binary review."""

    response: bool
    reviewer_model: str


class CritiqueResult(TypedDict):
    """Result from critique review."""

    response: str
    reviewer_model: str


class ReviewResponse(BaseModel):
    """Response model for all review types."""

    binary: dict[str, BinaryResult] | None = None
    critique: dict[str, CritiqueResult] | None = None
    status: ReviewStatus = ReviewStatus.SUCCESS


class ReviewConfig(BaseModel):
    """Configuration for a review type."""

    max_tokens: int
    prompt_template: str


ReviewResult = dict[str, BinaryResult]

REVIEW_CONFIGS: dict[ReviewType, ReviewConfig] = {
    ReviewType.BINARY: ReviewConfig(
        max_tokens=8,
        prompt_template=BINARY_REVIEW_PROMPT,
    ),
    ReviewType.CRITIQUE: ReviewConfig(
        max_tokens=512,
        prompt_template=CRITIQUE_REVIEW_PROMPT,
    ),
}


REVIEW_LLMS: dict[ReviewType, dict[str, BaseChatModel]] = {}


def get_review_llms(review_type: ReviewType = ReviewType.BINARY) -> Mapping[str, BaseChatModel]:
    """Get all review LLM instances."""
    global REVIEW_LLMS
    if review_type not in REVIEW_LLMS:
        REVIEW_LLMS[review_type] = {
            "gpt-4o": get_gpt_4o_llm(REVIEW_CONFIGS[review_type].max_tokens),
            "gpt-4o-mini": get_gpt_4o_mini_llm(REVIEW_CONFIGS[review_type].max_tokens),
            "gemini-15-flash": get_gemini_15_flash_llm(REVIEW_CONFIGS[review_type].max_tokens),
            "gemini-2-flash": get_gemini_2_flash_llm(REVIEW_CONFIGS[review_type].max_tokens),
        }
    return REVIEW_LLMS[review_type]


class BaseReviewLabeler(LLMLabeler[tuple[str, str], ReviewResultType], Generic[ReviewResultType]):
    """Base class for all review labelers."""

    def __init__(
        self,
        review_type: ReviewType,
        model: str = "gpt-4o",
        timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS,
    ) -> None:
        self.model = model
        self.review_type = review_type
        llms = get_review_llms(self.review_type)
        if self.model not in llms:
            logging.warning(f"{self.__class__.__name__}: Unsupported model {model}, using gpt-4o instead")
            self.model = "gpt-4o"
        self.base_llm = llms[self.model]
        super().__init__(self.base_llm, timeout_secs=timeout_secs)

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        """Prepare the LLM with the appropriate prompt template."""
        config = REVIEW_CONFIGS[self.review_type]
        prompt = fill_cur_datetime(config.prompt_template)
        template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                (
                    "human",
                    (
                        "Conversation History and Latest Query: {conversation_until_last_user_message}\n"
                        "AI Response to Evaluate: {response}"
                    ),
                ),
            ]
        )
        return template | llm  # type: ignore

    def _prepare_input(self, input: tuple[str, str]) -> dict[str, Any]:
        """Prepare input for the LLM."""
        return dict(
            conversation_until_last_user_message=input[0],
            response=input[1],
        )

    @property
    def error_value(self) -> ReviewResultType:
        raise NotImplementedError("Subclasses must implement this method")


class BinaryReviewLabeler(BaseReviewLabeler[bool]):
    """Labeler that determines if a response accurately answers the last human message with a binary true/false."""

    def __init__(self, model: str = "gpt-4o", timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS) -> None:
        super().__init__(ReviewType.BINARY, model, timeout_secs)

    def _parse_output(self, output: BaseMessage) -> bool:
        return str(output.content).strip().lower() == "true"

    @property
    def error_value(self) -> bool:
        return False


class CritiqueReviewLabeler(BaseReviewLabeler[str]):
    """Labeler that provides a short critique of the response."""

    def __init__(self, model: str = "gpt-4o", timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS) -> None:
        super().__init__(ReviewType.CRITIQUE, model, timeout_secs)

    def _parse_output(self, output: BaseMessage) -> str:
        parsed_output = str(output.content).replace("\n", " ").replace("Critique:", "").strip()
        for i in range(1, 4):
            parsed_output = parsed_output.replace(f"Line {i}: ", "").strip()
        return parsed_output

    @property
    def error_value(self) -> str:
        return "Error: Review failed"


# Singleton instances with model tracking
BINARY_REVIEWER: dict[str, BinaryReviewLabeler] = {}
CRITIQUE_REVIEWER: dict[str, CritiqueReviewLabeler] = {}


def get_binary_reviewer(model: str = "gpt-4o") -> BinaryReviewLabeler:
    """Get or create the binary reviewer instance for the specified model."""
    if model not in BINARY_REVIEWER:
        BINARY_REVIEWER[model] = BinaryReviewLabeler(model=model)
    return BINARY_REVIEWER[model]


def get_critique_reviewer(model: str = "gpt-4o") -> CritiqueReviewLabeler:
    """Get or create the critique reviewer instance for the specified model."""
    if model not in CRITIQUE_REVIEWER:
        CRITIQUE_REVIEWER[model] = CritiqueReviewLabeler(model=model)
    return CRITIQUE_REVIEWER[model]


def _extract_conversation_until_last_user_message(chat_history_messages: list[BaseMessage]) -> str:
    """
    Extract the conversation history until the last human message (inclusive).
    """
    # Find the last human message index
    last_human_idx = -1
    for i in range(len(chat_history_messages) - 1, -1, -1):
        if isinstance(chat_history_messages[i], HumanMessage):
            last_human_idx = i
            break

    if last_human_idx == -1:
        raise ValueError("No human messages found in chat history")

    # Get all history up to and including last human message as conversation history
    conversation_history = chat_history_messages[: last_human_idx + 1]
    conversation_parts = []
    turn = 1
    for i in range(0, len(conversation_history), 2):
        if i + 1 < len(conversation_history):
            # Complete turn with both user and system message
            user_msg = conversation_history[i]
            sys_msg = conversation_history[i + 1]
            conversation_parts.append(f"User Message (Turn {turn}): {user_msg.content}")
            conversation_parts.append(f"AI Response (Turn {turn}): {sys_msg.content}")
            conversation_parts.append("")  # Empty line between turns
            turn += 1
        else:
            # Final user message without response
            user_msg = conversation_history[i]
            conversation_parts.append(f"User Message (Turn {turn}): {user_msg.content}")

    conversation_until_last_user_message = "\n".join(conversation_parts)
    return conversation_until_last_user_message


async def generate_reviews(
    request: ReviewRequest,
) -> ReviewResponse:
    """Review responses using binary review.

    Args:
        request: The review request containing turn_id, review_types, etc.

    Returns:
        ReviewResponse: Object containing results for binary review
            - binary: dict[str, BinaryResult] if binary review was requested
    """

    async with get_async_session() as session:
        stmt = select(Turn.chat_id).where(Turn.turn_id == request.turn_id)  # type: ignore
        result = await session.execute(stmt)
        chat_id_result = result.scalar_one_or_none()
        if chat_id_result is None:
            raise ValueError(f"Turn {request.turn_id} not found")
        chat_id = str(chat_id_result)

    start_time = time.time()
    turn_id = UUID(request.turn_id) if request.turn_id else None
    chat_history = await get_curated_chat_context(
        chat_id=UUID(chat_id),
        use_all_models_in_chat_history=False,
        model=request.reviewer_model or "gpt-4o",
        current_turn_id=turn_id,
        context_for_logging="review",
        include_current_turn=True,
        return_all_current_turn_responses=True,
    )
    responses = chat_history.current_turn_responses

    if not responses:
        logging.warning(f"generate_reviews: No messages found for turn {request.turn_id}")
    if not chat_history.messages:
        logging.warning(f"generate_reviews: No chat history found for turn {request.turn_id}")
    chat_history_time = time.time() - start_time

    # Add attachments to the chat history
    attachments = [
        attachment for m in chat_history.messages for attachment in m.additional_kwargs.get("attachments", [])
    ]

    if len(attachments) > 0:
        logging.warning(
            f"generate_reviews: Attachments not implemented for review, "
            f"returning empty review response for turn {request.turn_id}"
        )
        return ReviewResponse(
            binary={},
            status=ReviewStatus.UNSUPPORTED,
        )

    # Extract conversation history until last user message from chat history
    conversation_until_last_user_message = _extract_conversation_until_last_user_message(chat_history.messages)
    last_assistant_responses = responses
    # Default to all review types if none specified
    review_types = request.review_types or list(ReviewType)

    # Initialize reviewer with specified model
    logging.info(f"Initializing reviewers with model {request.reviewer_model or 'gpt-4o'}")
    binary_reviewer = get_binary_reviewer(request.reviewer_model or "gpt-4o")
    critique_reviewer = get_critique_reviewer(request.reviewer_model or "gpt-4o")

    async def _run_pointwise_review(
        review_type: ReviewType,
        reviewer: BaseReviewLabeler[ReviewResultType],
    ) -> tuple[ReviewType, dict[str, BinaryResult | CritiqueResult]]:
        """Generic function to run pointwise reviews.

        Args:
            review_type: Type of review being performed
            reviewer: The reviewer instance to use

        Returns:
            Tuple of review type and results dictionary
        """
        assert isinstance(reviewer, BinaryReviewLabeler) or isinstance(
            reviewer, CritiqueReviewLabeler
        ), f"Reviewer {reviewer} is not a BinaryReviewLabeler or CritiqueReviewLabeler"
        model_name = reviewer.model

        try:
            # Run review for each response concurrently
            review_tasks = []
            if last_assistant_responses:
                for model, response in last_assistant_responses.items():
                    task = reviewer.alabel((conversation_until_last_user_message, response))
                    review_tasks.append((model, task))
            else:
                logging.warning(f"run_{review_type}: No responses to review, last_assistant_responses is empty")

            # Wait for all reviews to complete
            results = await asyncio.gather(*(task for _, task in review_tasks), return_exceptions=True)

            # Process results with proper type casting
            pointwise_results: dict[str, BinaryResult | CritiqueResult] = {}

            for (model, _), result in zip(review_tasks, results, strict=True):
                if isinstance(result, bool) or isinstance(result, str):  # Handle both bool and str results
                    if review_type == ReviewType.BINARY and isinstance(result, bool):
                        pointwise_results[model] = cast(
                            BinaryResult, {"response": result, "reviewer_model": model_name}
                        )
                    elif review_type == ReviewType.CRITIQUE and isinstance(result, str):
                        pointwise_results[model] = cast(
                            CritiqueResult, {"response": result, "reviewer_model": model_name}
                        )
                    else:
                        logging.warning(f"run_{review_type}: {model} returned {result} of type {type(result)}")
                else:
                    logging.warning(f"run_{review_type}: {model} returned {result} of type {type(result)}")

            return review_type, pointwise_results
        except Exception as e:
            log_dict = {
                "message": f"Error in {review_type} review for turn_id {request.turn_id}: {str(e)}",
                "conversation_until_last_user_message": conversation_until_last_user_message,
                "last_assistant_responses": last_assistant_responses,
            }
            logging.exception(json_dumps(log_dict))
            return review_type, {}

    async def run_binary_review() -> tuple[ReviewType, dict[str, BinaryResult]]:
        """Run binary review."""
        result = await _run_pointwise_review(
            ReviewType.BINARY,
            binary_reviewer,
        )
        return cast(tuple[ReviewType, dict[str, BinaryResult]], result)

    async def run_critique_review() -> tuple[ReviewType, dict[str, CritiqueResult]]:
        """Run critique review."""
        result = await _run_pointwise_review(
            ReviewType.CRITIQUE,
            critique_reviewer,
        )
        logging.info(f"Critique review result: {result}")
        return cast(tuple[ReviewType, dict[str, CritiqueResult]], result)

    # Map review types to their corresponding functions
    review_funcs = {
        ReviewType.BINARY: run_binary_review,
        ReviewType.CRITIQUE: run_critique_review,
    }

    # Map review types to their error values with proper type annotations
    error_values: dict[ReviewType, ReviewResult] = {review_type: {} for review_type in review_types}

    # Run selected review types in parallel
    tasks = [review_funcs[review_type]() for review_type in review_types]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and handle any exceptions
    processed_results: dict[ReviewType, ReviewResult] = {}
    has_error = False
    for review_result in zip(review_types, results, strict=True):
        review_type = review_result[0]
        result_or_exception = review_result[1]
        if isinstance(result_or_exception, tuple):
            processed_results[review_type] = cast(ReviewResult, result_or_exception[1])
        elif review_type in error_values:  # Only use error values for requested types
            has_error = True
            logging.warning(f"Error in {review_type}: {error_values[review_type]}")
            processed_results[review_type] = error_values[review_type]

    end_time = time.time()
    log_dict = {
        "message": "Reviews generated",
        "chat_history_time_ms": str(int(chat_history_time * 1000)),
        "chat_id": chat_id,
        "turn_id": request.turn_id,
        "reviewer_model": request.reviewer_model,
        "duration_secs": str(end_time - start_time),
        "review_types": [rt.value for rt in review_types],
    }
    logging.info(json_dumps(log_dict))
    logging.info(f"Processed results: {processed_results}")
    # Convert dict[ReviewType, ReviewResult] to ReviewResponse
    return ReviewResponse(
        binary=cast(dict[str, BinaryResult], processed_results.get(ReviewType.BINARY)),
        critique=cast(dict[str, CritiqueResult], processed_results.get(ReviewType.CRITIQUE)),
        status=ReviewStatus.ERROR if has_error else ReviewStatus.SUCCESS,
    )
