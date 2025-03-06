import logging
import time
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from fastapi import HTTPException, status
from sqlmodel import desc, select
from ypl.backend.db import get_async_session
from ypl.backend.llm.judge import FeedbackQualityLabeler
from ypl.backend.llm.labeler import MultiLLMLabeler
from ypl.backend.llm.provider.provider_clients import get_internal_provider_client
from ypl.backend.utils.json import json_dumps
from ypl.db.app_feedback import AppFeedback

FEEDBACK_QUALITY_JUDGING_TIMEOUT = 2
VERY_POOR_FEEDBACK_SCORE = 1
POOR_FEEDBACK_SCORE = 2
AVERAGE_FEEDBACK_SCORE = 3
GOOD_FEEDBACK_SCORE = 4
EXCELLENT_FEEDBACK_SCORE = 5
FALLBACK_QUALITY_SCORE = AVERAGE_FEEDBACK_SCORE


@dataclass
class FeedbackResponse:
    """Response model for a single feedback."""

    feedback_id: UUID
    user_comment: str
    created_at: datetime | None
    user_id: str
    chat_id: UUID


@dataclass
class FeedbacksResponse:
    """Response model for a paginated list of feedback."""

    feedbacks: list[FeedbackResponse]
    has_more_rows: bool


_MULTI_LABELER: MultiLLMLabeler | None = None


async def get_multi_labeler() -> MultiLLMLabeler:
    global _MULTI_LABELER

    if _MULTI_LABELER is None:
        _MULTI_LABELER = MultiLLMLabeler(
            labelers={
                "gpt4": FeedbackQualityLabeler(await get_internal_provider_client("gpt-4o-mini", max_tokens=16)),
                "gemini": FeedbackQualityLabeler(
                    await get_internal_provider_client("gemini-2.0-flash-001", max_tokens=16)
                ),
            },
            timeout_secs=FEEDBACK_QUALITY_JUDGING_TIMEOUT,
            early_terminate_on=["gpt4", "gemini"],
        )

    return _MULTI_LABELER


async def get_feedback_quality_score(user_id: str, feedback: str | None) -> int:
    """
    Evaluate the quality of user feedback using multiple LLM models.
    Uses MultiLLMLabeler to get the fastest response from available models.

    Args:
        user_id: The ID of the user providing feedback
        feedback: The feedback text to evaluate

    Returns:
        int: Quality score from 1-5, where:
            1 = Very Poor
            2 = Poor
            3 = Average
            4 = Good
            5 = Excellent
    """
    start_time = time.time()
    if not feedback:
        return POOR_FEEDBACK_SCORE

    try:
        multi_labeler = await get_multi_labeler()
        results = await multi_labeler.alabel(feedback)
        for model_name, result in results.items():
            if isinstance(result, int):
                elapsed_ms = (time.time() - start_time) * 1000
                log_dict = {
                    "message": "Feedback quality score latency",
                    "feedback": feedback,
                    "latency_ms": elapsed_ms,
                    "score": result,
                    "user_id": user_id,
                    "winning_model": model_name,
                }
                logging.info(json_dumps(log_dict))
                if result == -1:
                    log_dict = {
                        "message": "Error evaluating feedback quality",
                        "error": "Fallback quality score",
                        "user_id": user_id,
                        "feedback_length": len(feedback),
                    }
                    logging.warning(json_dumps(log_dict))
                else:
                    return result

        log_dict = {
            "message": "Timeout getting feedback quality score",
            "feedback": feedback,
            "user_id": user_id,
            "timeout": time.time() - start_time,
            "results": str(results),
        }
        logging.warning(json_dumps(log_dict))
        return FALLBACK_QUALITY_SCORE

    except Exception as e:
        log_dict = {
            "message": "Error evaluating feedback quality",
            "error": str(e),
            "user_id": user_id,
            "feedback_length": len(feedback),
        }
        logging.warning(json_dumps(log_dict))
        return FALLBACK_QUALITY_SCORE


async def store_app_feedback(app_feedback: AppFeedback) -> None:
    if app_feedback.quality_score is None:
        app_feedback.quality_score = await get_feedback_quality_score(
            str(app_feedback.user_id),
            app_feedback.user_comment,
        )
    async with get_async_session() as session:
        session.add(app_feedback)
        await session.commit()


async def get_paginated_feedback(
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> FeedbacksResponse:
    """Get app feedback with pagination support."""
    try:
        async with get_async_session() as session:
            query = select(AppFeedback).order_by(desc(AppFeedback.created_at))

            if user_id is not None:
                query = query.where(AppFeedback.user_id == user_id)

            query = query.offset(offset).limit(limit + 1)
            result = await session.execute(query)
            rows = result.scalars().all()

            has_more_rows = len(rows) > limit
            if has_more_rows:
                rows = rows[:-1]

            log_dict = {
                "message": "Admin: Feedback entries found",
                "user_id": user_id,
                "feedback_count": len(rows),
                "limit": limit,
                "offset": offset,
                "has_more_rows": has_more_rows,
            }
            logging.info(json_dumps(log_dict))

            return FeedbacksResponse(
                feedbacks=[
                    FeedbackResponse(
                        feedback_id=feedback.feedback_id,
                        user_comment=feedback.user_comment,
                        created_at=feedback.created_at,
                        user_id=feedback.user_id,
                        chat_id=feedback.chat_id,
                    )
                    for feedback in rows
                ],
                has_more_rows=has_more_rows,
            )

    except Exception as e:
        log_dict = {
            "message": "Admin: Unexpected error getting feedback",
            "user_id": user_id,
            "limit": limit,
            "offset": offset,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from e
