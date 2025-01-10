import json
import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from starlette.status import HTTP_400_BAD_REQUEST
from ypl.backend.utils.json import json_dumps

router = APIRouter(tags=["coinbase"])


@router.post("/webhook/{token}")
async def handle_coinbase_webhook(
    token: str, request: Request, x_coinbase_signature: str | None = Header(None, alias="X-Coinbase-Signature")
) -> dict[str, Any]:
    """Handle incoming webhooks from Coinbase.

    Args:
        token: The webhook token from the URL
        request: The incoming request
        x_coinbase_signature: The X-Coinbase-Signature header for webhook verification

    Returns:
        Dict[str, Any]: Response indicating success

    Raises:
        HTTPException: If request is invalid
    """
    if not x_coinbase_signature:
        log_dict = {
            "message": "Coinbase webhook signature is missing",
            "token": token,
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Bad request")

    try:
        payload = await request.body()
        event_data = json.loads(payload)

        message = f"*Coinbase Webhook Received*\n" f"Token: `{token}`\n" f"```{json.dumps(event_data, indent=2)}```"

        logging.info(message)

        return {"status": "success"}

    except json.JSONDecodeError as err:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Bad request") from err
    except Exception as err:
        logging.error(f"Error processing Coinbase webhook: {str(err)}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Bad request") from err
