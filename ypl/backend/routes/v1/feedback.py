import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ypl.backend.feedback.app_feedback import store_app_feedback
from ypl.backend.utils.json import json_dumps
from ypl.db.app_feedback import AppFeedback

router = APIRouter()


class AppFeedbackRequest(BaseModel):
    user_id: str
    chat_id: UUID | None = None
    user_comment: str


@router.post("/app_feedback")
async def log_app_feedback(request: AppFeedbackRequest) -> None:
    try:
        app_feedback = AppFeedback(
            user_id=request.user_id,
            chat_id=request.chat_id,
            user_comment=request.user_comment,
        )
        # store into database
        await store_app_feedback(app_feedback)

        # log
        comment_excerpt = request.user_comment[:100] if request.user_comment else None
        logging.info(
            json_dumps(
                {"message": f"App feedback: {comment_excerpt}", "user_id": request.user_id, "chat_id": request.chat_id}
            )
        )

    except Exception as e:
        log_dict = {"message": f"Error storing app feedback - {str(e)}"}
        logging.exception(json_dumps(log_dict))

        raise HTTPException(status_code=500, detail=str(e)) from e
