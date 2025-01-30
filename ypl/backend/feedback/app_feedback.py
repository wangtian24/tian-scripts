import logging
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from fastapi import HTTPException, status
from sqlmodel import desc, select
from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.app_feedback import AppFeedback


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


async def store_app_feedback(app_feedback: AppFeedback) -> None:
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
