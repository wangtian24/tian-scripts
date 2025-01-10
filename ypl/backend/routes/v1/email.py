import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ypl.backend.email.send_email import send_email_async
from ypl.backend.utils.json import json_dumps

router = APIRouter()


class SendEmailRequest(BaseModel):
    campaign: str
    to_address: str
    params: dict[str, str]


@router.post("/send_email")
async def send_email(request: SendEmailRequest) -> dict[str, Any]:
    try:
        email = await send_email_async(
            campaign=request.campaign,
            to_address=request.to_address,
            params=request.params,
        )
        return {"status": "success", "email": email}
    except ValueError as e:
        log_dict = {
            "message": f"Invalid email parameters: {e}",
            "campaign": request.campaign,
            "to_address": request.to_address,
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log_dict = {
            "message": f"Error sending email: {e}",
            "campaign": request.campaign,
            "to_address": request.to_address,
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
