import json
import logging

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.exc import NoResultFound

from ypl.backend.llm.credit import get_user_credit_balance, get_user_credit_balance_sync

router = APIRouter()


@router.get("/credits/balance")
async def get_credits_balance(user_id: str = Query(..., description="User ID")) -> int:
    try:
        return await get_user_credit_balance(user_id)

    except NoResultFound as e:
        raise HTTPException(status_code=404, detail="User not found") from e

    except Exception as e:
        log_dict = {
            "message": "Error getting credit balance",
            "error": str(e),
        }
        logging.exception(json.dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


# Temporary sync endpoint to test performance of async vs sync.
@router.get("/credits/balance-sync")
async def get_credits_balance_sync(user_id: str = Query(..., description="User ID")) -> int:
    try:
        return get_user_credit_balance_sync(user_id)

    except NoResultFound as e:
        raise HTTPException(status_code=404, detail="User not found") from e

    except Exception as e:
        log_dict = {
            "message": "Error getting credit balance",
            "error": str(e),
        }
        logging.exception(json.dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
