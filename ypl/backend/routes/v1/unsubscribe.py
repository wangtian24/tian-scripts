import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ypl.backend.llm.unsubscribe import unsubscribe_from_marketing_emails

router = APIRouter()


class UnsubscribeRequest(BaseModel):
    user_id: str


class UnsubscribeResponse(BaseModel):
    message: str


@router.post("/unsubscribe", response_model=UnsubscribeResponse)
async def unsubscribe(request: UnsubscribeRequest) -> UnsubscribeResponse:
    try:
        await unsubscribe_from_marketing_emails(request.user_id)
        return UnsubscribeResponse(message="Unsubscribed from marketing emails")
    except ValueError as e:
        logging.error(f"Invalid user ID {request.user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid user ID") from e
    except Exception as e:
        logging.error(f"Error unsubscribing user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
