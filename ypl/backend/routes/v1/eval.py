import logging

from fastapi import APIRouter, HTTPException

from ypl.backend.feedback.eval import EvalRequest, store_message_eval
from ypl.backend.utils.json import json_dumps

router = APIRouter()


@router.post("/eval/message")
async def create_message_eval_route(request: EvalRequest) -> None:
    try:
        await store_message_eval(request)
    except Exception as e:
        log_dict = {"message": f"Error storing app feedback to database - {str(e)}"}
        logging.exception(json_dumps(log_dict))

        raise HTTPException(status_code=500, detail=str(e)) from e
