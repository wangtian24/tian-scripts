"""Type definitions and models for review functionality."""

import enum
import uuid
from typing import TypeVar

from pydantic import BaseModel
from typing_extensions import TypedDict

from ypl.backend.config import settings

# Type variable for review results
ReviewResultType = TypeVar("ReviewResultType", bool, str, list[dict[str, str]])


class ReviewType(str, enum.Enum):
    """Type of review to perform."""

    BINARY = "binary"
    CRITIQUE = "critique"
    SEGMENTED = "segmented"
    NUGGETIZED = "nuggetized"


class ReviewStatus(str, enum.Enum):
    """Status of the review operation."""

    SUCCESS = "success"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class ReviewRequest(BaseModel):
    """Request model for review operations."""

    turn_id: uuid.UUID | None = None
    review_types: list[ReviewType] | None = None
    fallback_reviewer_model_name: str | None = None
    reviewer_model_preference: list[str] | None = None
    timeout_secs: float = settings.DEFAULT_REVIEW_TIMEOUT_SECS


class BinaryResult(TypedDict):
    """Result from binary review."""

    response: bool
    reviewer_model: str


class CritiqueResult(TypedDict):
    """Result from critique review."""

    response: str
    reviewer_model: str


class SegmentedResult(TypedDict):
    """Result from segmented review."""

    segments: list[dict[str, str]]  # List of segments with their updates and reviews
    reviewer_model: str


class NuggetizedResult(TypedDict):
    """Result from nuggetized review."""

    nuggets: list[dict[str, str]]  # List of nuggets with text, importance, and assignment
    reviewer_model: str


class ReviewResponse(BaseModel):
    """Response model for all review types."""

    binary: dict[str, BinaryResult] | None = None
    critique: dict[str, CritiqueResult] | None = None
    segmented: dict[str, SegmentedResult] | None = None
    nuggetized: dict[str, NuggetizedResult] | None = None
    status: ReviewStatus


class ReviewConfig(BaseModel):
    """Configuration for a review type."""

    max_tokens: int
    prompt_template: str


AllReviewResults = BinaryResult | CritiqueResult | SegmentedResult | NuggetizedResult
ReviewResult = dict[str, AllReviewResults]
