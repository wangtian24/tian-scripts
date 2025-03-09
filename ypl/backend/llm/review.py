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
from ypl.backend.llm.context import get_curated_chat_context
from ypl.backend.llm.labeler import LLMLabeler
from ypl.backend.llm.nuggetizer import YuppNuggetizer
from ypl.backend.llm.provider.provider_clients import (
    get_internal_provider_client,
)
from ypl.backend.llm.review_types import (
    BinaryResult,
    CritiqueResult,
    CrossCheckResult,
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
from ypl.backend.prompts.reviews import (
    BINARY_REVIEW_PROMPT,
    BINARY_REVIEW_USER_PROMPT,
    CRITIQUE_REVIEW_PROMPT,
    CRITIQUE_REVIEW_USER_PROMPT,
    CROSS_CHECK_PROMPT,
    CROSS_CHECK_USER_PROMPT,
    SEGMENTED_REVIEW_PROMPT,
    SEGMENTED_REVIEW_USER_PROMPT,
)
from ypl.backend.prompts.system_prompts import fill_cur_datetime, partial_format
from ypl.backend.utils.json import json_dumps
from ypl.db.chats import ChatMessage, Turn
from ypl.db.language_models import LanguageModel
from ypl.db.reviews import MessageReview

REVIEW_CONFIGS: dict[ReviewType, ReviewConfig] = {
    ReviewType.BINARY: ReviewConfig(
        max_tokens=8,
        prompt_template=BINARY_REVIEW_PROMPT,
        user_prompt_template=BINARY_REVIEW_USER_PROMPT,
    ),
    ReviewType.CRITIQUE: ReviewConfig(
        max_tokens=512,
        prompt_template=CRITIQUE_REVIEW_PROMPT,
        user_prompt_template=CRITIQUE_REVIEW_USER_PROMPT,
    ),
    ReviewType.SEGMENTED: ReviewConfig(
        max_tokens=4096,  # Larger token limit for detailed segmented reviews
        prompt_template=SEGMENTED_REVIEW_PROMPT,
        user_prompt_template=SEGMENTED_REVIEW_USER_PROMPT,
    ),
    ReviewType.NUGGETIZED: ReviewConfig(
        max_tokens=4096,  # Larger token limit for nuggetized reviews
        prompt_template="",  # No prompt template needed as we use YuppNuggetizer
    ),
    ReviewType.CROSS_CHECK: ReviewConfig(
        max_tokens=1024,
        prompt_template=CROSS_CHECK_PROMPT,
        user_prompt_template=CROSS_CHECK_USER_PROMPT,
    ),
}

REVIEW_MODEL_WITH_PDF_SUPPORT = ["gemini-2.0-flash-001"]

# Whitelist of models supported for reviews
REVIEW_MODEL_ALLOWLIST_PREFERENCES = [
    "gpt-4o",
    "gemini-2.0-flash-001",
    "claude-3-7-sonnet-20250219",
    "qwen-max-2025-01-25",  # TODO(ronak): Verify for segmented reviews when the time comes
    "deepseek/deepseek-chat",  # TODO(ronak): Verify for segmented reviews when the time comes
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "qwen2.5-vl-72b-instruct",
]

REVIEW_MODEL_WITH_IMAGE_ALLOWLIST_PREFERENCES = [
    "gpt-4o",
    "gemini-2.0-flash-001",
    "claude-3-7-sonnet-20250219",
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "qwen2.5-vl-72b-instruct",
]

REVIEW_MODEL_FALLBACK = "gpt-4o"


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


class BaseReviewLabeler(LLMLabeler[tuple[list[BaseMessage], str], ReviewResultType], Generic[ReviewResultType]):
    """Base class for all review labelers."""

    def __init__(
        self,
        review_type: ReviewType,
        model: str,
        llm: BaseChatModel,
        timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS,
        chat_history: list[BaseMessage] | None = None,
    ) -> None:
        self.model = model
        self.review_type = review_type
        self.base_llm = llm
        self.chat_history = chat_history or []
        super().__init__(self.base_llm, timeout_secs=timeout_secs)

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        """Prepare the LLM with the appropriate prompt template."""
        config = REVIEW_CONFIGS[self.review_type]
        system_prompt = fill_cur_datetime(config.prompt_template)
        user_prompt = config.user_prompt_template

        # Extract conversation until last user message
        conversation_history: list[BaseMessage] = (
            _extract_conversation_until_last_user_message(self.chat_history) if self.chat_history else []
        )

        # Create prompt messages
        messages: list[tuple[str, str] | BaseMessage] = [("system", system_prompt)]

        # Add conversation history
        messages += conversation_history

        # Add AI response evaluation prompt
        messages.append(("human", "AI Response to Evaluate: {response}\n\n"))

        messages.append(("human", user_prompt))

        template = ChatPromptTemplate.from_messages(messages)
        return template | llm  # type: ignore

    def _prepare_input(self, input: tuple[list[BaseMessage], str]) -> dict[str, Any]:
        """Prepare input for the LLM."""
        chat_history, response = input

        # If chat_history is provided in init, use it; otherwise use the one from input
        if self.chat_history and not chat_history:
            # Already processed in _prepare_llm
            return dict(response=response)
        else:
            # Update chat_history for future calls
            self.chat_history = chat_history
            return dict(response=response)


class BinaryReviewLabeler(BaseReviewLabeler[bool]):
    """Labeler that determines if a response accurately answers the last human message with a binary true/false."""

    def __init__(
        self,
        model: str,
        llm: BaseChatModel,
        timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS,
        chat_history: list[BaseMessage] | None = None,
    ) -> None:
        super().__init__(ReviewType.BINARY, model, llm, timeout_secs, chat_history)

    def _parse_output(self, output: BaseMessage) -> bool:
        return str(output.content).strip().lower() == "true"


class CritiqueReviewLabeler(BaseReviewLabeler[str]):
    """Labeler that provides a short critique of the response."""

    def __init__(
        self,
        model: str,
        llm: BaseChatModel,
        timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS,
        chat_history: list[BaseMessage] | None = None,
    ) -> None:
        super().__init__(ReviewType.CRITIQUE, model, llm, timeout_secs, chat_history)

    def _parse_output(self, output: BaseMessage) -> str:
        parsed_output = str(output.content).replace("\n", " ").replace("Critique:", "").strip()
        for i in range(1, 4):
            parsed_output = parsed_output.replace(f"Line {i}: ", "").strip()
        return parsed_output


class SegmentedReviewLabeler(BaseReviewLabeler[list[dict[str, str]]]):
    """Labeler that provides segmented review with updates and reasoning."""

    def __init__(
        self,
        model: str,
        llm: BaseChatModel,
        timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS,
        chat_history: list[BaseMessage] | None = None,
    ) -> None:
        super().__init__(ReviewType.SEGMENTED, model, llm, timeout_secs, chat_history)

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


class NuggetizedReviewLabeler(LLMLabeler[tuple[list[BaseMessage], list[tuple[str, str]]], dict[str, NuggetizedResult]]):
    """Labeler that uses nuggetizer to extract and analyze key points from responses."""

    def __init__(
        self, model: str, llm: BaseChatModel, timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS
    ) -> None:
        self.model = model
        self.base_llm = llm
        self.nuggetizer: YuppNuggetizer | None = None
        super().__init__(self.base_llm, timeout_secs=timeout_secs)

    def _get_nuggetizer(self) -> YuppNuggetizer:
        if self.nuggetizer is None:
            self.nuggetizer = YuppNuggetizer(model=self.model)
        return self.nuggetizer

    async def alabel_full(
        self, input: tuple[list[BaseMessage], list[tuple[str, str]]]
    ) -> tuple[dict[str, NuggetizedResult], str]:
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

                # Convert question list into a single string
                question_text = ""
                for i, q in enumerate(question):
                    if i % 2 == 0:
                        question_text += f"User Message (Turn {i // 2 + 1}): "
                    else:
                        question_text += f"AI Response (Turn {i // 2 + 1}): "
                    if isinstance(q, str):
                        question_text += f"{q}\n"
                    elif isinstance(q, list) and q and isinstance(q[-1], dict):
                        question_text += str(q[-1].get("text", "")) + "\n"
                # Create nuggetizer request
                query = Query(qid="review-tmp-qid", text=question_text.strip())
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

    def _prepare_input(self, input: tuple[list[BaseMessage], list[tuple[str, str]]]) -> dict[str, Any]:
        """Input is (question, list of (model_name, response)) tuple"""
        question, model_responses = input
        return dict(question=question, model_responses=model_responses)

    async def _aparse_output(self, output: BaseMessage) -> dict[str, NuggetizedResult]:
        """Parse output from nuggetizer. This is a no-op since alabel_full handles everything."""
        return {}

    @property
    def error_value(self) -> dict[str, NuggetizedResult]:
        return {}


class CrossCheckReviewLabeler(LLMLabeler[tuple[list[BaseMessage], str, str], str]):
    """Labeler that provides a cross-check review comparing a model's response with other models' responses."""

    def __init__(
        self,
        model: str,
        llm: BaseChatModel,
        timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS,
        chat_history: list[BaseMessage] | None = None,
        other_model_names: str | None = None,
    ) -> None:
        self.model = model
        self.review_type = ReviewType.CROSS_CHECK
        self.base_llm = llm
        self.chat_history = chat_history or []
        self.other_model_names = other_model_names
        super().__init__(self.base_llm, timeout_secs=timeout_secs)

    def _prepare_llm(self, llm: BaseChatModel) -> BaseChatModel:
        """Prepare the LLM with the appropriate prompt template."""
        config = REVIEW_CONFIGS[self.review_type]
        prompt = fill_cur_datetime(config.prompt_template)
        # Apply partial formatting for other_model_names
        prompt = partial_format(prompt, other_model_names=self.other_model_names)
        user_prompt = config.user_prompt_template

        # Extract conversation until last user message
        conversation_history: list[BaseMessage] = (
            _extract_conversation_until_last_user_message(self.chat_history) if self.chat_history else []
        )

        # Create prompt messages
        messages: list[tuple[str, str] | BaseMessage] = [("system", prompt)]

        # Add conversation history
        messages += conversation_history
        # Add other assistant's response to evaluate
        final_human_message = "Other Assistants' Responses to Evaluate: {other_response}\n\n"
        final_human_message += "Your previous response to the user's prompt was: {your_response}\n\n"
        final_human_message += user_prompt
        messages.append(("human", final_human_message))

        template = ChatPromptTemplate.from_messages(messages)
        return template | llm  # type: ignore

    def _prepare_input(self, input: tuple[list[BaseMessage], str, str]) -> dict[str, Any]:
        """Prepare input for the LLM.

        Args:
            input: Tuple of (chat_history, your_response, other_response)
        """
        chat_history, your_response, other_response = input

        if not self.chat_history:
            self.chat_history = chat_history
        return dict(your_response=your_response, other_response=other_response)

    def _parse_output(self, output: BaseMessage) -> str:
        """Parse the output from the LLM."""
        return str(output.content).strip()

    @property
    def error_value(self) -> str:
        return ""


BINARY_REVIEWER: dict[str, BinaryReviewLabeler] = {}
CRITIQUE_REVIEWER: dict[str, CritiqueReviewLabeler] = {}
SEGMENTED_REVIEWER: dict[str, SegmentedReviewLabeler] = {}
NUGGETIZED_REVIEWER: dict[str, NuggetizedReviewLabeler] = {}
CROSS_CHECK_REVIEWER: dict[str, CrossCheckReviewLabeler] = {}


async def get_reviewer(
    review_type: ReviewType,
    model: str = REVIEW_MODEL_FALLBACK,
    chat_history: list[BaseMessage] | None = None,
    other_model_names: str | None = None,
) -> BaseReviewLabeler[ReviewResultType]:
    """Get a reviewer instance for the specified review type and model.

    Args:
        review_type: The type of review to perform
        model: The model to use for the review (defaults to REVIEW_MODEL_FALLBACK)
        chat_history: Optional chat history to provide to the reviewer
        other_model_names: Names of other models for cross-check review (only used with CROSS_CHECK)

    Returns:
        A reviewer instance of the appropriate type
    """
    # Validate model and use fallback if needed
    if (
        review_type not in [ReviewType.NUGGETIZED, ReviewType.CROSS_CHECK]
        and model not in REVIEW_MODEL_ALLOWLIST_PREFERENCES
    ):
        logging.warning(f"{review_type} reviewer: Unsupported model {model}, using {REVIEW_MODEL_FALLBACK} instead")
        model = REVIEW_MODEL_FALLBACK
    elif review_type == ReviewType.NUGGETIZED and model not in ["gpt-4o", "gpt-4o-mini"]:
        logging.warning(f"NuggetizedReviewLabeler: Unsupported model {model}, using {REVIEW_MODEL_FALLBACK} instead")
        model = REVIEW_MODEL_FALLBACK

    # Get LLM client (already cached by get_internal_provider_client)
    max_tokens = REVIEW_CONFIGS[review_type].max_tokens
    llm = await get_internal_provider_client(model, max_tokens=max_tokens)

    # Create appropriate reviewer instance
    if review_type == ReviewType.NUGGETIZED:
        return cast(
            BaseReviewLabeler[ReviewResultType],
            NuggetizedReviewLabeler(model=model, llm=llm),
        )
    elif review_type == ReviewType.CROSS_CHECK:
        return cast(
            BaseReviewLabeler[ReviewResultType],
            CrossCheckReviewLabeler(
                model=model, llm=llm, chat_history=chat_history, other_model_names=other_model_names
            ),
        )

    reviewer_class = {
        ReviewType.BINARY: BinaryReviewLabeler,
        ReviewType.CRITIQUE: CritiqueReviewLabeler,
        ReviewType.SEGMENTED: SegmentedReviewLabeler,
    }[review_type]
    return cast(BaseReviewLabeler[ReviewResultType], reviewer_class(model=model, llm=llm, chat_history=chat_history))


def _extract_conversation_until_last_user_message(
    chat_history_messages: list[BaseMessage],
) -> list[BaseMessage]:
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
    return chat_history_messages[: last_human_idx + 1]


def _select_reviewer_model(
    response_model_family: str | None,
    reviewer_model_preference: list[str] | None,
    default_model: str = REVIEW_MODEL_FALLBACK,
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
        model=request.fallback_reviewer_model_name or REVIEW_MODEL_FALLBACK,
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
    has_pdf_attachments = any(attachment.content_type == "application/pdf" for attachment in attachments)
    has_image_attachments = any(
        attachment.content_type is not None and attachment.content_type.startswith("image/")
        for attachment in attachments
    )
    parse_pdf_locally = settings.PARSE_PDF_LOCALLY_FOR_REVIEW
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
    last_assistant_responses = responses
    last_assistant_response_models = chat_history.current_turn_models
    # Default to all review types if none specified
    review_types = request.review_types or list(ReviewType)

    # Get response_model_family_map and review_llms_model_family_map from DB
    all_models: set[str] = set()
    if responses:
        all_models.update(responses.keys())
        all_models.update(REVIEW_MODEL_ALLOWLIST_PREFERENCES)  # Add whitelist models for family lookup

    if all_models:
        all_model_families = await get_model_families_and_ids(list(all_models))
        # Extract just the family part for the existing logic
        response_model_family_map = (
            {k: v[0] for k, v in all_model_families.items() if k in responses} if responses else {}
        )
        review_llms_model_family_map = {
            model: all_model_families[model][0]
            for model in REVIEW_MODEL_ALLOWLIST_PREFERENCES
            if model in all_model_families
        }
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
            reviewer_model_preference = request.reviewer_model_preference or [
                "gpt-4o",
                "gemini-2.0-flash-001",
                "claude-3-7-sonnet-20250219",
                "qwen-max-2025-01-25",
                "deepseek/deepseek-chat",
            ]
            fallback_reviewer_model_name_default = request.fallback_reviewer_model_name or REVIEW_MODEL_FALLBACK
            if has_pdf_attachments and not parse_pdf_locally:
                reviewer_model_preference = request.reviewer_model_preference or REVIEW_MODEL_WITH_PDF_SUPPORT
                fallback_reviewer_model_name_default = (
                    reviewer_model_preference[0] if reviewer_model_preference else REVIEW_MODEL_FALLBACK
                )
            elif has_image_attachments:
                reviewer_model_preference = (
                    request.reviewer_model_preference or REVIEW_MODEL_WITH_IMAGE_ALLOWLIST_PREFERENCES
                )
                fallback_reviewer_model_name_default = (
                    reviewer_model_preference[0] if reviewer_model_preference else REVIEW_MODEL_FALLBACK
                )
            reviewer_model = _select_reviewer_model(
                model_family,
                reviewer_model_preference,
                fallback_reviewer_model_name_default,
                review_llms_model_family_map,
            )

            if ReviewType.BINARY in review_types:
                binary_reviewers[model] = await get_reviewer(ReviewType.BINARY, reviewer_model, chat_history.messages)
            if ReviewType.CRITIQUE in review_types:
                critique_reviewers[model] = await get_reviewer(
                    ReviewType.CRITIQUE, reviewer_model, chat_history.messages
                )
            if ReviewType.SEGMENTED in review_types:
                segmented_reviewers[model] = await get_reviewer(
                    ReviewType.SEGMENTED, reviewer_model, chat_history.messages
                )

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
                    task = reviewer.alabel((chat_history.messages, response.content))
                    review_tasks.append((model, task))
            else:
                logging.warning(f"run_{review_type}: No responses to review, last_assistant_responses is empty")

            results = await asyncio.gather(*(task for _, task in review_tasks), return_exceptions=True)

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
                "chat_history": chat_history.messages,
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
            reviewer_model = REVIEW_MODEL_FALLBACK  # Default to REVIEW_MODEL_FALLBACK (gpt-4o) for nuggetized review
            nuggetized_reviewer = cast(
                NuggetizedReviewLabeler, await get_reviewer(ReviewType.NUGGETIZED, reviewer_model)
            )

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

    async def run_cross_check_review() -> tuple[ReviewType, dict[str, CrossCheckResult]]:
        """Run cross check review."""
        try:
            cross_check_results: dict[str, CrossCheckResult] = {}

            if last_assistant_responses is None or len(last_assistant_responses.keys()) < 2:
                logging.warning(" Need at least 2 responses to compare with cross-check")
                return ReviewType.CROSS_CHECK, {}

            # Get conversation history until last user message
            conversation_history = _extract_conversation_until_last_user_message(chat_history.messages)

            # Prepare all cross-check tasks
            review_tasks = []
            for model_name, response in last_assistant_responses.items():
                other_model_responses = {
                    other_model: other_response
                    for other_model, other_response in last_assistant_responses.items()
                    if other_model != model_name
                }
                # Skip if no other models to compare with
                if not other_model_responses:
                    continue
                if not last_assistant_response_models:
                    continue
                # Build combined response of other models
                other_models_text = "\n---\n".join(
                    [
                        f"Response from {last_assistant_response_models.get(other_model, other_model)}:\n{r.content}"
                        for other_model, r in other_model_responses.items()
                    ]
                )
                other_model_names = ", ".join(
                    [
                        last_assistant_response_models.get(other_model, other_model)
                        for other_model in other_model_responses.keys()
                    ]
                )
                # Default to model_name for cross check review
                reviewer_model = model_name

                # Create reviewer and run task
                cross_check_reviewer = cast(
                    CrossCheckReviewLabeler,
                    await get_reviewer(ReviewType.CROSS_CHECK, reviewer_model, conversation_history, other_model_names),
                )

                # Add task to the list
                task = cross_check_reviewer.alabel((conversation_history, response.content, other_models_text))
                review_tasks.append((model_name, reviewer_model, other_model_names, task))

            # Run all tasks in parallel
            results = await asyncio.gather(*(task for _, _, _, task in review_tasks), return_exceptions=True)

            # Process results
            for (model_name, reviewer_model, other_model_names, _), result in zip(review_tasks, results, strict=True):
                if isinstance(result, Exception):
                    logging.warning(f"Error in cross check review: {str(result)}")
                    continue
                cross_check_results[model_name] = {
                    "response": result if isinstance(result, str) else str(result),
                    "reviewer_model": reviewer_model,
                    "other_model_names": other_model_names,
                }

            return ReviewType.CROSS_CHECK, cross_check_results

        except Exception as e:
            log_dict = {
                "message": f"Error in cross check review: {str(e)}",
                "question": _extract_conversation_until_last_user_message(chat_history.messages),
                "last_assistant_responses": last_assistant_responses,
            }
            logging.exception(json_dumps(log_dict))
            return ReviewType.CROSS_CHECK, {}

    review_funcs = {
        ReviewType.BINARY: run_binary_review,
        ReviewType.CRITIQUE: run_critique_review,
        ReviewType.SEGMENTED: run_segmented_review,
        ReviewType.NUGGETIZED: run_nuggetized_review,
        ReviewType.CROSS_CHECK: run_cross_check_review,
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
        "reviewer_model": request.fallback_reviewer_model_name,
        "duration_secs": str(end_time - start_time),
        "review_types": [rt.value for rt in review_types],
    }
    logging.info(json_dumps(log_dict))

    # Persist reviews to database
    if request.turn_id is not None:
        if last_assistant_responses:
            # Collect all reviewers that were used
            all_reviewers: dict[ReviewType, dict[str, BaseReviewLabeler[Any]]] = {
                ReviewType.BINARY: binary_reviewers,
                ReviewType.CRITIQUE: critique_reviewers,
                ReviewType.SEGMENTED: segmented_reviewers,
            }

            # Add nuggetized reviewers if they were used
            if ReviewType.NUGGETIZED in review_types and ReviewType.NUGGETIZED in processed_results:
                nuggetized_reviewer_dict: dict[str, BaseReviewLabeler[Any]] = {}
                # For each model with a nuggetized result, store the reviewer that was used
                for model_name in processed_results[ReviewType.NUGGETIZED].keys():
                    if model_name in last_assistant_responses:
                        # In the nuggetized case, we use a fixed reviewer model
                        nuggetized_reviewer = await get_reviewer(ReviewType.NUGGETIZED, REVIEW_MODEL_FALLBACK)
                        nuggetized_reviewer_dict[model_name] = cast(BaseReviewLabeler[Any], nuggetized_reviewer)
                all_reviewers[ReviewType.NUGGETIZED] = nuggetized_reviewer_dict

            # Add cross check reviewers if they were used
            if ReviewType.CROSS_CHECK in review_types and ReviewType.CROSS_CHECK in processed_results:
                cross_check_reviewer_dict: dict[str, BaseReviewLabeler[Any]] = {}
                # For each comparison key with a cross check result, store the reviewer that was used
                for model_name in processed_results[ReviewType.CROSS_CHECK].keys():
                    # In the cross check case, we use a fixed reviewer model
                    cross_check_reviewer = await get_reviewer(ReviewType.CROSS_CHECK, model_name, None, None)
                    cross_check_reviewer_dict[model_name] = cast(BaseReviewLabeler[Any], cross_check_reviewer)
                all_reviewers[ReviewType.CROSS_CHECK] = cross_check_reviewer_dict

            await store_reviews(
                processed_results=processed_results,
                last_assistant_responses=last_assistant_responses,
                review_types=review_types,
                reviewers=all_reviewers,
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
        cross_check=cast(dict[str, CrossCheckResult], processed_results.get(ReviewType.CROSS_CHECK)),
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
                        continue

                    # Special handling for nuggetized reviews
                    if review_type == ReviewType.NUGGETIZED:
                        # Get the nuggetized result for this model
                        review_result = processed_results[review_type].get(model_name)
                        if not review_result:
                            logging.warning(f"Nuggetized review result for {model_name} not found in processed results")
                            continue

                        # For nuggetized reviews, we use a fixed reviewer model (gpt-4o)
                        reviewer_model = REVIEW_MODEL_FALLBACK
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
                        continue

                    if review_type not in reviewers or model_name not in reviewers[review_type]:
                        logging.warning(
                            f"Reviewer for {model_name} and {review_type} not found in reviewers "
                            f"(message_id={message.message_id}, turn_id={message.turn_id}, chat_id={chat_id})"
                        )
                        continue

                    reviewer = reviewers[review_type][model_name]
                    reviewer_model = reviewer.model
                    if reviewer_model not in model_info:
                        logging.warning(
                            f"Reviewer model {reviewer_model} not found in model_info "
                            f"(message_id={message.message_id}, turn_id={message.turn_id}, chat_id={chat_id})"
                        )
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

            await session.commit()

    except Exception as e:
        logging.error(
            f"Error storing reviews to DB: {e}. "
            f"Failed messages: {[m.message_id for m in last_assistant_responses.values()]} "
            f"chat_id: {chat_id}"
        )
