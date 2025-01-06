import json
import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import httpx
from ypl.backend.config import settings
from ypl.backend.payment.base_types import PaymentProcessingError
from ypl.backend.payment.upi.axis.cryptography_utils import (
    aes128_decrypt,
    aes128_encrypt,
    calculate_request_body_checksum,
)

RequestType = Literal["get_balance", "verify_vpa", "make_payment", "get_payment_status", "get_account_statement"]


@dataclass
class Request:
    request_id: str
    url: str
    request_type: RequestType
    encrypted_request: dict
    plaintext_request: dict


def _get_config_value(key: str, request_type: RequestType | None = None) -> str:
    config = settings.axis_upi_config
    if request_type and key in config[request_type]:
        return config[request_type][key]  # type: ignore[no-any-return]
    if key not in config:
        raise ValueError(f"Missing required config key: {key}")
    return config[key]  # type: ignore[no-any-return]


def _log_request(request: Request) -> None:
    log_dict = {
        "message": f"Calling Axis {request.request_type} API with URL: {request.url}",
        "request_id": request.request_id,
        "encrypted_request": request.encrypted_request,
        "plaintext_request": request.plaintext_request,
    }
    logging.info(json.dumps(log_dict))


def _log_response_body(request: Request, response_body: dict) -> None:
    log_dict = {
        "message": f"Decrypted response body from Axis {request.request_type} API",
        "request_id": request.request_id,
        "response_body": response_body,
    }
    logging.info(json.dumps(log_dict))


async def _call(request: Request) -> dict:
    _log_request(request)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            request.url,
            headers={
                "X-IBM-Client-Id": _get_config_value("client_id"),
                "X-IBM-Client-Secret": _get_config_value("client_secret"),
            },
            json=request.encrypted_request,
        )

        response_json = None
        try:
            response_json = response.json()
            logging.info(
                json.dumps(
                    {
                        "message": f"Received response from Axis {request.request_type} API",
                        "status_code": response.status_code,
                        "response": response_json,
                        "request_id": request.request_id,
                    }
                )
            )
            response.raise_for_status()
            return response_json  # type: ignore[no-any-return]
        except httpx.HTTPStatusError as e:
            logging.error(
                json.dumps(
                    {
                        "message": f"HTTP error in Axis {request.request_type} API",
                        "status_code": response.status_code,
                        "response": response_json,
                        "error": str(e),
                        "request_id": request.request_id,
                    }
                )
            )
            raise


def _make_get_balance_request() -> Request:
    request_id = str(uuid.uuid4())
    request_body = {
        "corpCode": _get_config_value("corp_code", "get_balance"),
        "corpAccNum": _get_config_value("account_number", "get_balance"),
        "channelId": _get_config_value("channel_id", "get_balance"),
    }
    request_body["checksum"] = calculate_request_body_checksum(request_body)

    sub_header = {
        "requestUUID": request_id,
        "serviceRequestId": "OpenAPI",
        "serviceRequestVersion": "1.0",
        "channelId": _get_config_value("channel_id"),
    }

    encrypted_request = {
        "GetAccountBalanceRequest": {
            "SubHeader": sub_header,
            "GetAccountBalanceRequestBodyEncrypted": aes128_encrypt(
                _get_config_value("aes_symmetric_key"), json.dumps(request_body)
            ),
        }
    }

    plaintext_request = {
        "GetAccountBalanceRequest": {"SubHeader": sub_header, "GetAccountBalanceRequestBody": request_body}
    }

    return Request(
        request_id,
        _get_config_value("url", "get_balance"),
        "get_balance",
        encrypted_request,
        plaintext_request,
    )


async def get_balance() -> Decimal:
    request = _make_get_balance_request()
    response = await _call(request)
    decrypted_body = aes128_decrypt(
        _get_config_value("aes_symmetric_key"),
        response["GetAccountBalanceResponse"]["GetAccountBalanceResponseBodyEncrypted"],
    )
    json_body = json.loads(decrypted_body)
    _log_response_body(request, json_body)
    if json_body["status"] != "S":
        raise PaymentProcessingError("Failed to get balance")
    return Decimal(json_body["data"]["Balance"])
