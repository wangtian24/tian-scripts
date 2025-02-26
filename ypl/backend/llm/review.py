"""Module for reviewing message accuracy."""

import asyncio
import logging
import time
from typing import Any, Generic, cast
from uuid import UUID

from async_lru import alru_cache
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from nuggetizer.core.types import Document, Query, Request
from sqlalchemy import select

from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.llm.chat import get_curated_chat_context
from ypl.backend.llm.labeler import LLMLabeler
from ypl.backend.llm.nuggetizer import YuppNuggetizer
from ypl.backend.llm.provider.provider_clients import (
    get_gemini_2_flash_llm,
    get_gemini_15_flash_llm,
    get_gpt_4o_llm,
    get_gpt_4o_mini_llm,
)
from ypl.backend.llm.review_types import (
    BinaryResult,
    CritiqueResult,
    NuggetizedResult,
    ReviewConfig,
    ReviewRequest,
    ReviewResponse,
    ReviewResult,
    ReviewResultType,
    ReviewStatus,
    ReviewType,
    SegmentedResult,
)
from ypl.backend.llm.transform_messages import TransformOptions, transform_user_messages
from ypl.backend.prompts import (
    BINARY_REVIEW_PROMPT,
    CRITIQUE_REVIEW_PROMPT,
    SEGMENTED_REVIEW_PROMPT,
    fill_cur_datetime,
)
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import ChatMessage, Turn
from ypl.db.language_models import LanguageModel
from ypl.db.reviews import MessageReview

REVIEW_CONFIGS: dict[ReviewType, ReviewConfig] = {
    ReviewType.BINARY: ReviewConfig(
        max_tokens=8,
        prompt_template=BINARY_REVIEW_PROMPT,
    ),
    ReviewType.CRITIQUE: ReviewConfig(
        max_tokens=512,
        prompt_template=CRITIQUE_REVIEW_PROMPT,
    ),
    ReviewType.SEGMENTED: ReviewConfig(
        max_tokens=4096,  # Larger token limit for detailed segmented reviews
        prompt_template=SEGMENTED_REVIEW_PROMPT,
    ),
    ReviewType.NUGGETIZED: ReviewConfig(
        max_tokens=4096,  # Larger token limit for nuggetized reviews
        prompt_template="",  # No prompt template needed as we use YuppNuggetizer
    ),
}

REVIEW_MODEL_WITH_PDF_SUPPORT = ["gemini-2.0-flash-exp"]


@alru_cache(maxsize=None, ttl=86400)  # 24-hour cache, now async-compatible
async def get_model_family_and_id(model_name: str) -> tuple[str, UUID]:
    """Get model family and ID from database with 24-hour caching."""
    async with get_async_session() as session:
        try:
            stmt = select(LanguageModel.family, LanguageModel.language_model_id).where(  # type: ignore
                LanguageModel.internal_name == model_name
            )
            result = await session.execute(stmt)
            row = result.first()
            if not row:
                logging.warning(f"Model {model_name} not found")
                raise ValueError(f"Model {model_name} not found")
            family, model_id = row
            if not family or not model_id:
                logging.warning(f"Model {model_name} not found")
                raise ValueError(f"Model {model_name} not found")
            return str(family), model_id
        except Exception as e:
            logging.exception(f"Error getting model info for {model_name}: {e}")
            raise e


async def get_model_families_and_ids(model_names: list[str]) -> dict[str, tuple[str, UUID]]:
    """Get model families and IDs from database by looking up each model individually."""
    return {model_name: await get_model_family_and_id(model_name) for model_name in model_names}


REVIEW_LLMS: dict[ReviewType, dict[str, BaseChatModel]] = {}


# TODO(Tian): This probably needs some refactoring to use get_internal_provider_client(), but it's not
# an async function due to the way BaseReviewLabeler is constructed. This needs some bigger refactoring
# to make all get_xxx_reviewer() functions async and pass LLM clients from outside rather than
# having each labeler class getting its own LLM clients. Keep as is for now.
def get_review_llms(review_type: ReviewType = ReviewType.BINARY) -> dict[str, BaseChatModel]:
    """Get all review LLM instances."""
    global REVIEW_LLMS
    if review_type not in REVIEW_LLMS:
        REVIEW_LLMS[review_type] = {
            "gpt-4o": get_gpt_4o_llm(REVIEW_CONFIGS[review_type].max_tokens),
            "gpt-4o-mini": get_gpt_4o_mini_llm(REVIEW_CONFIGS[review_type].max_tokens),
            "gemini-1.5-flash-002": get_gemini_15_flash_llm(REVIEW_CONFIGS[review_type].max_tokens),
            "gemini-2.0-flash-exp": get_gemini_2_flash_llm(REVIEW_CONFIGS[review_type].max_tokens),
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


class BinaryReviewLabeler(BaseReviewLabeler[bool]):
    """Labeler that determines if a response accurately answers the last human message with a binary true/false."""

    def __init__(self, model: str = "gpt-4o", timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS) -> None:
        super().__init__(ReviewType.BINARY, model, timeout_secs)

    def _parse_output(self, output: BaseMessage) -> bool:
        return str(output.content).strip().lower() == "true"


class CritiqueReviewLabeler(BaseReviewLabeler[str]):
    """Labeler that provides a short critique of the response."""

    def __init__(self, model: str = "gpt-4o", timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS) -> None:
        super().__init__(ReviewType.CRITIQUE, model, timeout_secs)

    def _parse_output(self, output: BaseMessage) -> str:
        parsed_output = str(output.content).replace("\n", " ").replace("Critique:", "").strip()
        for i in range(1, 4):
            parsed_output = parsed_output.replace(f"Line {i}: ", "").strip()
        return parsed_output


class SegmentedReviewLabeler(BaseReviewLabeler[list[dict[str, str]]]):
    """Labeler that provides segmented review with updates and reasoning."""

    def __init__(self, model: str = "gpt-4o", timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS) -> None:
        super().__init__(ReviewType.SEGMENTED, model, timeout_secs)

    def _process_tag_content(self, lines: list[str], start_idx: int, tag: str) -> tuple[str, int]:
        """Process content for a specific XML-style tag in the segmented review format.

        This method extracts content between XML-style tags of the format:
        <tag N>
        content lines...
        </tag N>

        where N is a numeric identifier. The method handles accidental generation cases like [insert verbatim ...]
        markers and preserves all whitespace/formatting in the content.

        Args:
            lines: List of lines to process from the LLM output
            start_idx: Current line index pointing to the opening tag line
            tag: Tag name (one of: "segment", "review", or "updated-segment")

        Returns:
            tuple[str, int]: A tuple containing:
                - The extracted content as a single string with newlines preserved
                - The index of the last processed line (the closing tag line)

        Example:
            For input lines:
                <segment 1>
                This is some content
                 across multiple lines
                </segment 1>

            With start_idx=0 and tag="segment", returns:
            ("This is some content\n across multiple lines", 3)
        """
        tag_num = lines[start_idx].strip()[len(f"<{tag}") + 1 : -1].strip()
        content = []
        i = start_idx + 1
        while i < len(lines):
            if lines[i].strip() == f"</{tag} {tag_num}>":
                break
            if lines[i].strip().startswith("[insert verbatim"):
                i += 1
                continue
            content.append(lines[i])
            i += 1
        return "\n".join(content), i

    def _parse_output(self, output: BaseMessage) -> list[dict[str, str]]:
        content = str(output.content)
        segments = []
        current_segment = {}
        lines = content.split("\n")
        i = 0

        while i < len(lines):
            if i == 0 and lines[0].strip() == "...":
                i += 1
                continue

            line = lines[i].strip()
            if line.startswith("<segment"):
                current_segment["segment"], i = self._process_tag_content(lines, i, "segment")
            elif line.startswith("<review"):
                current_segment["review"], i = self._process_tag_content(lines, i, "review")
            elif line.startswith("<updated-segment"):
                current_segment["update"], i = self._process_tag_content(lines, i, "updated-segment")
                segments.append(current_segment)
                current_segment = {}
            i += 1

        return segments


class NuggetizedReviewLabeler(LLMLabeler[tuple[str, list[tuple[str, str]]], dict[str, NuggetizedResult]]):
    """Labeler that uses nuggetizer to extract and analyze key points from responses."""

    def __init__(self, model: str = "gpt-4o", timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS) -> None:
        self.model = model
        if self.model not in ["gpt-4o", "gpt-4o-mini"]:
            logging.warning(f"NuggetizedReviewLabeler: Unsupported model {model}, using gpt-4o instead")
            self.model = "gpt-4o"
        self.base_llm = get_gpt_4o_llm(max_tokens=4096)
        self.nuggetizer: YuppNuggetizer | None = None
        super().__init__(self.base_llm, timeout_secs=timeout_secs)

    def _get_nuggetizer(self) -> YuppNuggetizer:
        if self.nuggetizer is None:
            self.nuggetizer = YuppNuggetizer(model=self.model)
        return self.nuggetizer

    async def alabel_full(self, input: tuple[str, list[tuple[str, str]]]) -> tuple[dict[str, NuggetizedResult], str]:
        """Labels the input asynchronously using nuggetizer.

        Args:
            input: A tuple containing:
                - question: The conversation history until the last user message
                - model_responses: List of tuples (model_name, response_text)

        Returns:
            A tuple containing:
                - model_specific_results: Dict mapping model names to their NuggetizedResult
                - empty string (no raw output to clean)
        """
        try:
            start_time = time.time()
            async with asyncio.timeout(self.timeout_secs):
                question, model_responses = input
                prepared_input = self._prepare_input(input)

                # Extract model names and responses
                model_names = [model_name for model_name, _ in model_responses]
                responses = [response for _, response in model_responses]

                if not responses:
                    logging.warning("No responses to review in nuggetizer")
                    return {}, ""

                # Create nuggetizer request
                query = Query(qid="review-tmp-qid", text=question)
                documents = [
                    Document(docid=f"review-tmp-response-{i+1}", segment=response)
                    for i, response in enumerate(responses)
                ]
                request = Request(query=query, documents=documents)

                # Initialize nuggetizer
                nuggetizer = self._get_nuggetizer()

                # Extract nuggets
                scored_nuggets = await nuggetizer.create(request)

                # Assign nuggets to each response in parallel
                assignment_tasks = [nuggetizer.assign(query.text, response, scored_nuggets) for response in responses]
                assigned_nuggets_list = await asyncio.gather(*assignment_tasks)

                # Create model-specific results
                model_specific_results: dict[str, NuggetizedResult] = {}

                # Process each model's results
                for i, model_name in enumerate(model_names):
                    if i < len(assigned_nuggets_list):
                        model_nuggets = [
                            {
                                "text": nugget.text,
                                "importance": nugget.importance,
                                "assignment": assigned_nuggets_list[i][j].assignment,
                            }
                            for j, nugget in enumerate(scored_nuggets)
                        ]

                        model_specific_results[model_name] = {
                            "nuggets": model_nuggets,
                            "reviewer_model": self.model,
                        }

                return model_specific_results, ""  # Empty string as we don't have raw output to clean
        except Exception as e:
            self._log_error(input, prepared_input, e, start_time)
            if self.on_error == "raise":
                raise e
            else:
                return {}, ""

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        return llm  # No prompt template needed for nuggetizer

    def _prepare_input(self, input: tuple[str, list[tuple[str, str]]]) -> dict[str, Any]:
        """Input is (question, list of (model_name, response)) tuple"""
        question, model_responses = input
        return dict(question=question, model_responses=model_responses)

    async def _aparse_output(self, output: BaseMessage) -> dict[str, NuggetizedResult]:
        """Parse output from nuggetizer. This is a no-op since alabel_full handles everything."""
        return {}

    @property
    def error_value(self) -> dict[str, NuggetizedResult]:
        return {}


# Singleton instances with model tracking
BINARY_REVIEWER: dict[str, BinaryReviewLabeler] = {}
CRITIQUE_REVIEWER: dict[str, CritiqueReviewLabeler] = {}
SEGMENTED_REVIEWER: dict[str, SegmentedReviewLabeler] = {}
NUGGETIZED_REVIEWER: dict[str, NuggetizedReviewLabeler] = {}


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


def get_segmented_reviewer(model: str = "gpt-4o") -> SegmentedReviewLabeler:
    """Get or create the segmented reviewer instance for the specified model."""
    if model not in SEGMENTED_REVIEWER:
        SEGMENTED_REVIEWER[model] = SegmentedReviewLabeler(model=model)
    return SEGMENTED_REVIEWER[model]


def get_nuggetized_reviewer(model: str = "gpt-4o") -> NuggetizedReviewLabeler:
    """Get or create the nuggetized reviewer instance."""
    if model not in NUGGETIZED_REVIEWER:
        NUGGETIZED_REVIEWER[model] = NuggetizedReviewLabeler(model)
    return NUGGETIZED_REVIEWER[model]


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


def _select_reviewer_model(
    response_model_family: str | None,
    reviewer_model_preference: list[str] | None,
    default_model: str = "gpt-4o",
    review_llms_model_family_map: dict[str, str] | None = None,
) -> str:
    """Select appropriate reviewer model based on response model family, preferences, and review LLMs model family map.

    Args:
        response_model_family: Family of the model that generated the response
        reviewer_model_preference: Ordered list of preferred reviewer models
        default_model: Default model to use if no preference matches
        review_llms_model_family_map: Map of review LLM model to its family

    Returns:
        Selected reviewer model name
    """
    if not response_model_family or not reviewer_model_preference:
        return default_model

    # Find first preferred model from a different family
    for model in reviewer_model_preference:
        if (
            review_llms_model_family_map
            and model in review_llms_model_family_map
            and review_llms_model_family_map[model] != response_model_family
        ):
            return model

    return default_model


async def generate_reviews(
    request: ReviewRequest,
) -> ReviewResponse:
    """Review responses using binary review.

    Args:
        request: The review request containing turn_id, review_types, etc.

    Returns:
        ReviewResponse: Object containing results for binary review
            - binary: dict[str, BinaryResult] if binary review was requested
            - critique: dict[str, CritiqueResult] if critique review was requested
            - segmented: dict[str, SegmentedResult] if segmented review was requested
            - nuggetized: dict[str, NuggetizedResult] if nuggetized review was requested
    """
    async with get_async_session() as session:
        stmt = select(Turn.chat_id).where(Turn.turn_id == request.turn_id)  # type: ignore
        result = await session.execute(stmt)
        chat_id_result = result.scalar_one_or_none()
        if chat_id_result is None:
            raise ValueError(f"Turn {request.turn_id} not found")
        chat_id = str(chat_id_result)
    start_time = time.time()
    turn_id = request.turn_id
    chat_history = await get_curated_chat_context(
        chat_id=UUID(chat_id),
        use_all_models_in_chat_history=False,
        model=request.fallback_reviewer_model_name or "gpt-4o",
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
    has_attachments = len(attachments) > 0
    has_pdf_attachments = False
    parse_pdf_locally = settings.PARSE_PDF_LOCALLY_FOR_REVIEW
    if has_attachments:
        has_pdf_attachments = any(attachment.content_type == "application/pdf" for attachment in attachments)
        transform_options: TransformOptions = {
            "image_type": "thumbnail",
            "use_signed_url": False,
            "parse_pdf_locally": parse_pdf_locally,
            "max_pdf_text": settings.MAX_TEXT_TO_EXTRACT_FROM_PDF,
        }
        chat_history.messages = await transform_user_messages(
            chat_history.messages, REVIEW_MODEL_WITH_PDF_SUPPORT[0], options=transform_options
        )
    # Extract conversation history until last user message from chat history
    conversation_until_last_user_message = _extract_conversation_until_last_user_message(chat_history.messages)
    last_assistant_responses = responses
    # Default to all review types if none specified
    review_types = request.review_types or list(ReviewType)
    review_llms = get_review_llms()
    # Get response_model_family_map and review_llms_model_family_map from DB
    all_models: set[str] = set()
    if responses:
        all_models.update(responses.keys())
    if review_llms:
        all_models.update(review_llms.keys())

    if all_models:
        all_model_families = await get_model_families_and_ids(list(all_models))
        # Extract just the family part for the existing logic
        response_model_family_map = (
            {k: v[0] for k, v in all_model_families.items() if k in responses} if responses else {}
        )
        review_llms_model_family_map = (
            {k: v[0] for k, v in all_model_families.items() if k in review_llms} if review_llms else {}
        )
    else:
        response_model_family_map = {}
        review_llms_model_family_map = {}

    # Create reviewers for each response model based on family
    binary_reviewers: dict[str, BaseReviewLabeler[bool]] = {}
    critique_reviewers: dict[str, BaseReviewLabeler[str]] = {}
    segmented_reviewers: dict[str, BaseReviewLabeler[list[dict[str, str]]]] = {}

    if last_assistant_responses:
        for model in last_assistant_responses.keys():
            model_family = response_model_family_map.get(model)
            reviewer_model_preference = request.reviewer_model_preference or ["gpt-4o", "gemini-2.0-flash-exp"]
            fallback_reviewer_model_name_default = request.fallback_reviewer_model_name or "gpt-4o"
            if has_pdf_attachments and not parse_pdf_locally:
                reviewer_model_preference = REVIEW_MODEL_WITH_PDF_SUPPORT
                fallback_reviewer_model_name_default = REVIEW_MODEL_WITH_PDF_SUPPORT[0]
            reviewer_model = _select_reviewer_model(
                model_family,
                reviewer_model_preference,
                fallback_reviewer_model_name_default,
                review_llms_model_family_map,
            )
            binary_reviewers[model] = get_binary_reviewer(reviewer_model)
            critique_reviewers[model] = get_critique_reviewer(reviewer_model)
            segmented_reviewers[model] = get_segmented_reviewer(reviewer_model)

    async def _run_pointwise_review(
        review_type: ReviewType,
        reviewers: dict[str, BaseReviewLabeler[ReviewResultType]],
    ) -> tuple[ReviewType, dict[str, BinaryResult | CritiqueResult | SegmentedResult]]:
        """Generic function to run pointwise reviews.

        Args:
            review_type: Type of review being performed
            reviewers: Dict mapping response model to its reviewer

        Returns:
            Tuple of review type and results dictionary
        """
        for reviewer in reviewers.values():
            assert isinstance(
                reviewer, (BinaryReviewLabeler | CritiqueReviewLabeler | SegmentedReviewLabeler)
            ), f"Reviewer {reviewer} is not a valid review labeler type"
        try:
            # Run review for each response concurrently
            review_tasks = []
            if last_assistant_responses:
                for model, response in last_assistant_responses.items():
                    reviewer = reviewers[model]
                    task = reviewer.alabel((conversation_until_last_user_message, response.content))
                    review_tasks.append((model, task))
            else:
                logging.warning(f"run_{review_type}: No responses to review, last_assistant_responses is empty")

            # Wait for all reviews to complete
            results = await asyncio.gather(*(task for _, task in review_tasks), return_exceptions=True)

            # Process results with proper type casting
            pointwise_results: dict[str, BinaryResult | CritiqueResult | SegmentedResult] = {}

            for (model, _), result in zip(review_tasks, results, strict=True):
                if isinstance(result, bool) or isinstance(result, str):  # Handle both bool and str results
                    if review_type == ReviewType.BINARY and isinstance(result, bool):
                        pointwise_results[model] = cast(
                            BinaryResult, {"response": result, "reviewer_model": reviewers[model].model}
                        )
                    elif review_type == ReviewType.CRITIQUE and isinstance(result, str):
                        pointwise_results[model] = cast(
                            CritiqueResult, {"response": result, "reviewer_model": reviewers[model].model}
                        )
                    else:
                        logging.warning(f"run_{review_type}: {model} returned {result} of type {type(result)}")
                elif isinstance(result, list):  # Check for list type first
                    if review_type == ReviewType.SEGMENTED and all(isinstance(x, dict) for x in result):
                        pointwise_results[model] = cast(
                            SegmentedResult, {"segments": result, "reviewer_model": reviewers[model].model}
                        )
                    else:
                        logging.warning(f"run_{review_type}: {model} returned invalid list result {result}")
                else:
                    logging.warning(f"run_{review_type}: {model} returned {result} of type {type(result)}")

            return review_type, pointwise_results
        except Exception as e:
            log_dict = {
                "message": f"_run_{review_type}: Error in {review_type} review for turn_id {request.turn_id}: {str(e)}",
                "conversation_until_last_user_message": conversation_until_last_user_message,
                "last_assistant_responses": last_assistant_responses,
            }
            logging.exception(json_dumps(log_dict))
            return review_type, {}

    async def run_binary_review() -> tuple[ReviewType, dict[str, BinaryResult]]:
        """Run binary review."""
        result = await _run_pointwise_review(
            ReviewType.BINARY,
            binary_reviewers,
        )
        return cast(tuple[ReviewType, dict[str, BinaryResult]], result)

    async def run_critique_review() -> tuple[ReviewType, dict[str, CritiqueResult]]:
        """Run critique review."""
        result = await _run_pointwise_review(
            ReviewType.CRITIQUE,
            critique_reviewers,
        )
        return cast(tuple[ReviewType, dict[str, CritiqueResult]], result)

    async def run_segmented_review() -> tuple[ReviewType, dict[str, SegmentedResult]]:
        """Run segmented review."""
        result = await _run_pointwise_review(
            ReviewType.SEGMENTED,
            segmented_reviewers,
        )
        return cast(tuple[ReviewType, dict[str, SegmentedResult]], result)

    async def run_nuggetized_review() -> tuple[ReviewType, dict[str, NuggetizedResult]]:
        """Run nuggetized review."""
        try:
            # Prepare model-specific responses
            model_responses: list[tuple[str, str]] = []
            if last_assistant_responses:
                for model_name, response in last_assistant_responses.items():
                    model_responses.append((model_name, str(response.content)))
            else:
                logging.warning("run_nuggetized_review: No responses to review, last_assistant_responses is empty")
                return ReviewType.NUGGETIZED, {}

            # Get conversation history until last user message
            conversation_history = _extract_conversation_until_last_user_message(chat_history.messages)

            # Use all responses for nuggetized review
            reviewer_model = "gpt-4o"  # Default to gpt-4o for nuggetized review
            nuggetized_reviewer = get_nuggetized_reviewer(reviewer_model)

            # Run the nuggetizer with model-specific responses
            model_specific_results, _ = await nuggetized_reviewer.alabel_full((conversation_history, model_responses))
            return ReviewType.NUGGETIZED, model_specific_results

        except Exception as e:
            log_dict = {
                "message": f"Error in nuggetized review: {str(e)}",
                "question": _extract_conversation_until_last_user_message(chat_history.messages),
                "last_assistant_responses": last_assistant_responses,
            }
            logging.exception(json_dumps(log_dict))
            return ReviewType.NUGGETIZED, {}

    # Map review types to their corresponding functions
    review_funcs = {
        ReviewType.BINARY: run_binary_review,
        ReviewType.CRITIQUE: run_critique_review,
        ReviewType.SEGMENTED: run_segmented_review,
        ReviewType.NUGGETIZED: run_nuggetized_review,
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
        logging.info(f"Review result or exception: {result_or_exception}")
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
        "reviewer_model": request.fallback_reviewer_model_name,
        "duration_secs": str(end_time - start_time),
        "review_types": [rt.value for rt in review_types],
    }
    logging.info(json_dumps(log_dict))

    # Persist reviews to database
    if request.turn_id is not None:
        if last_assistant_responses:
            await store_reviews(
                processed_results=processed_results,
                last_assistant_responses=last_assistant_responses,
                review_types=review_types,
                reviewers={
                    ReviewType.BINARY: binary_reviewers,
                    ReviewType.CRITIQUE: critique_reviewers,
                    ReviewType.SEGMENTED: segmented_reviewers,
                    ReviewType.NUGGETIZED: {
                        model: cast(BaseReviewLabeler[Any], get_nuggetized_reviewer("gpt-4o"))
                        for model in last_assistant_responses
                    }
                    if last_assistant_responses
                    else {},
                },
                has_error=has_error,
                model_info=all_model_families,
                chat_id=chat_id,
            )
    else:
        logging.warning("Cannot persist reviews: turn_id is None")

    # Convert dict[ReviewType, ReviewResult] to ReviewResponse
    return ReviewResponse(
        binary=cast(dict[str, BinaryResult], processed_results.get(ReviewType.BINARY)),
        critique=cast(dict[str, CritiqueResult], processed_results.get(ReviewType.CRITIQUE)),
        segmented=cast(dict[str, SegmentedResult], processed_results.get(ReviewType.SEGMENTED)),
        nuggetized=cast(dict[str, NuggetizedResult], processed_results.get(ReviewType.NUGGETIZED)),
        status=ReviewStatus.ERROR if has_error else ReviewStatus.SUCCESS,
    )


async def store_reviews(
    processed_results: dict[ReviewType, ReviewResult],
    last_assistant_responses: dict[str, ChatMessage],
    review_types: list[ReviewType],
    reviewers: dict[ReviewType, dict[str, BaseReviewLabeler[Any]]],
    has_error: bool,
    model_info: dict[str, tuple[str, UUID]],
    chat_id: str,
) -> None:
    """Persist generated reviews to the database.

    Args:
        processed_results: The review results by type
        last_assistant_responses: Map of model name to corresponding ChatMessage for the current turn
        review_types: List of review types that were generated
        reviewers: Map of review type to model->reviewer mapping
        has_error: Whether there was an error during review generation
        model_info: Map of model name to tuple of (family, model_id)
        chat_id: Chat ID
    """
    try:
        async with get_async_session() as session:
            # Process each message and review type
            for model_name, message in last_assistant_responses.items():
                for review_type in review_types:
                    if review_type not in processed_results:
                        logging.warning(
                            f"Review type {review_type} not found in processed results "
                            f"(message_id={message.message_id}, turn_id={message.turn_id}, chat_id={chat_id})"
                        )

                    # Special handling for nuggetized reviews
                    if review_type == ReviewType.NUGGETIZED:
                        # Get the nuggetized result for this model
                        review_result = processed_results[review_type].get(model_name)
                        if not review_result:
                            logging.warning(f"Nuggetized review result for {model_name} not found in processed results")
                            continue

                        # For nuggetized reviews, we use a fixed reviewer model (gpt-4o)
                        reviewer_model = "gpt-4o"
                        if reviewer_model not in model_info:
                            logging.warning(f"Reviewer model {reviewer_model} not found in model_info")
                            continue

                        _, reviewer_model_id = model_info[reviewer_model]

                        review = MessageReview(
                            message_id=message.message_id,
                            review_type=review_type,
                            result=review_result,
                            reviewer_model_id=reviewer_model_id,
                            status=ReviewStatus.ERROR if has_error else ReviewStatus.SUCCESS,
                        )
                        session.add(review)
                        continue

                    # Standard handling for other review types
                    review_result = processed_results[review_type].get(model_name)
                    if not review_result:
                        logging.warning(
                            f"Review result for {model_name} and {review_type} not found in processed results "
                            f"(message_id={message.message_id}, turn_id={message.turn_id}, chat_id={chat_id})"
                        )

                    reviewer = reviewers[review_type].get(model_name)
                    if not reviewer:
                        logging.warning(
                            f"Reviewer for {model_name} and {review_type} not found in reviewers "
                            f"(message_id={message.message_id}, turn_id={message.turn_id}, chat_id={chat_id})"
                        )
                        continue

                    if reviewer.model not in model_info:
                        logging.warning(
                            f"Reviewer model {reviewer.model} not found in model_info "
                            f"(message_id={message.message_id}, turn_id={message.turn_id}, chat_id={chat_id})"
                        )
                        continue

                    _, reviewer_model_id = model_info[reviewer.model]

                    review = MessageReview(
                        message_id=message.message_id,
                        review_type=review_type,
                        result=review_result,
                        reviewer_model_id=reviewer_model_id,
                        status=ReviewStatus.ERROR if has_error else ReviewStatus.SUCCESS,
                    )
                    session.add(review)

            await session.commit()

    except Exception as e:
        logging.error(
            f"Error storing reviews to DB: {e}. "
            f"Failed messages: {[m.message_id for m in last_assistant_responses.values()]} "
            f"chat_id: {chat_id}"
        )
