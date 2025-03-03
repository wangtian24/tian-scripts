import asyncio
import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ypl.backend.abuse.activity import SHORT_TIME_WINDOWS, check_activity_volume_abuse
from ypl.backend.feedback.app_feedback import FeedbacksResponse, get_paginated_feedback, store_app_feedback
from ypl.backend.utils.json import json_dumps
from ypl.db.app_feedback import AppFeedback

router = APIRouter()


class AppFeedbackRequest(BaseModel):
    user_id: str
    chat_id: UUID | None = None
    turn_id: UUID | None = None
    user_comment: str


@router.post("/app_feedback")
async def log_app_feedback(request: AppFeedbackRequest) -> None:
    try:
        app_feedback = AppFeedback(
            user_id=request.user_id,
            chat_id=request.chat_id,
            turn_id=request.turn_id,
            user_comment=request.user_comment,
        )
        # store into database
        await store_app_feedback(app_feedback)

        # log
        comment_excerpt = request.user_comment[:100] if request.user_comment else None
        logging.info(
            json_dumps(
                {
                    "message": f"App feedback: {comment_excerpt}",
                    "user_id": request.user_id,
                    "chat_id": request.chat_id,
                    "turn_id": request.turn_id,
                }
            )
        )
        asyncio.create_task(check_activity_volume_abuse(request.user_id, time_windows=SHORT_TIME_WINDOWS))

    except Exception as e:
        log_dict = {"message": f"Error storing app feedback to database - {str(e)}"}
        logging.exception(json_dumps(log_dict))

        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/app_feedback")
async def get_app_feedback(
    user_id: Annotated[str | None, Query(description="Optional User ID to filter messages")] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Number of messages to return")] = 50,
    offset: Annotated[int, Query(ge=0, description="Number of messages to skip")] = 0,
) -> FeedbacksResponse:
    return await get_paginated_feedback(user_id=user_id, limit=limit, offset=offset)
