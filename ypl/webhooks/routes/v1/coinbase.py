import asyncio
import hmac
import json
import logging
from hashlib import sha256
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from starlette.status import HTTP_400_BAD_REQUEST
from ypl.backend.config import settings
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps

router = APIRouter(tags=["coinbase"])

SLACK_WEBHOOK_CASHOUT = settings.SLACK_WEBHOOK_CASHOUT


def validate_payload(payload: bytes, webhook_id: str, x_coinbase_signature: str) -> None:
    """Validate the Coinbase webhook payload using HMAC-SHA256.

    Args:
        payload: The raw request body
        webhook_id: The webhook ID from the payload
        x_coinbase_signature: The X-Coinbase-Signature header value

    Raises:
        HTTPException: If payload validation fails
    """

    #  return incase of local environment
    if settings.ENVIRONMENT != "production":
        return

    try:
        # Concatenate webhookId and payload as bytes
        message = webhook_id.encode() + payload

        # Generate hash of the concatenated message
        calculated_signature = sha256(message).hexdigest()
        print(calculated_signature)
        if not hmac.compare_digest(calculated_signature, x_coinbase_signature):
            log_dict = {
                "message": "Invalid Coinbase webhook signature",
                "webhook_id": webhook_id,
                "x_coinbase_signature": x_coinbase_signature,
                "calculated_signature": calculated_signature,
            }
            logging.warning(json_dumps(log_dict))
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Invalid signature")

    except Exception as err:
        log_dict = {
            "message": "Error validating Coinbase webhook payload",
            "error": str(err),
        }
        logging.warning(json_dumps(log_dict))
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Invalid payload") from err


def validate_signature(token: str, x_coinbase_signature: str | None) -> None:
    """Validate the Coinbase webhook signature header is present.

    Args:
        token: The webhook token from the URL
        x_coinbase_signature: The X-Coinbase-Signature header for webhook verification

    Raises:
        HTTPException: If signature is missing or invalid
    """
    if not x_coinbase_signature:
        log_dict = {
            "message": "Coinbase webhook signature is missing",
            "token": token,
        }
        logging.warning(json_dumps(log_dict))
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Bad request")


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
    validate_signature(token, x_coinbase_signature)
    assert x_coinbase_signature is not None

    try:
        payload = await request.body()
        event_data = json.loads(payload)

        log_dict = {
            "message": "Coinbase webhook received",
            "event_data": event_data,
        }
        logging.info(json_dumps(log_dict))

        webhook_id = event_data.get("webhook_id")
        if not webhook_id:
            log_dict = {
                "message": "Missing webhook_id",
                "event_data": event_data,
            }
            logging.warning(json_dumps(log_dict))
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Missing webhook_id")

        validate_payload(payload, webhook_id, x_coinbase_signature)

        log_dict = {
            "message": "Coinbase webhook validated",
            "token": token,
            "event_data": event_data,
        }
        logging.info(json_dumps(log_dict))
        asyncio.create_task(post_to_slack(json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT))

        return {"status": "success"}

    except json.JSONDecodeError as err:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Bad request") from err
    except Exception as err:
        logging.error(f"Error processing Coinbase webhook: {str(err)}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Bad request") from err
