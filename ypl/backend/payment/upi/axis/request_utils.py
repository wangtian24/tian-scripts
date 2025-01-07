import base64
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Literal

import httpx
from ypl.backend.config import settings
from ypl.backend.payment.base_types import PaymentProcessingError, PaymentResponse
from ypl.backend.payment.upi.axis.cryptography_utils import (
    aes128_decrypt,
    aes128_encrypt,
    calculate_request_body_checksum,
)
from ypl.db.payments import PaymentTransactionStatusEnum

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


# Axis requires a unique ID that's below 30 characters in alphanumeric.
# We plan to use the base32 encoded UUID, with the padding removed and converted to lowercase.
def _uuid_to_axis_unique_id(uuid: uuid.UUID) -> str:
    return base64.b32encode(uuid.bytes).decode("utf-8").replace("=", "").lower()


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


@dataclass
class AxisPaymentRequest:
    internal_payment_transaction_id: uuid.UUID
    amount: Decimal
    # Will be passed as beneCode (beneficiary code), which can be any unique ID for the beneficiary.
    # We plan to use the base32 encoded UUID of the user's instrument ID.
    destination_internal_id: uuid.UUID
    # VPA or the VPA mapped phone number.
    destination_upi_id: str
    # Message to be displayed to the receiver.
    receiver_display_message: str


def _make_payment_request(axis_payment_request: AxisPaymentRequest) -> Request:
    request_id = str(uuid.uuid4())

    beneficiary_name_or_code = _uuid_to_axis_unique_id(axis_payment_request.destination_internal_id)

    request_body = {
        "channelId": _get_config_value("channel_id", "make_payment"),
        "corpCode": _get_config_value("corp_code", "make_payment"),
        "paymentDetails": [
            {
                "corpAccNum": _get_config_value("account_number", "make_payment"),
                # For UPI, the payment mode is always UP.
                "txnPaymode": "UP",
                "custUniqRef": _uuid_to_axis_unique_id(axis_payment_request.internal_payment_transaction_id),
                # Current date in YYYY-MM-DD format in IST.
                "valueDate": datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d"),
                "txnAmount": str(axis_payment_request.amount),
                "beneName": beneficiary_name_or_code,
                "beneCode": beneficiary_name_or_code,
                # For UPI, the beneficiary account number is the VPA.
                "beneAccNum": axis_payment_request.destination_upi_id,
                "senderToReceiverInfo": axis_payment_request.receiver_display_message,
            }
        ],
    }
    request_body["checksum"] = calculate_request_body_checksum(request_body)

    sub_header = {
        "requestUUID": request_id,
        "serviceRequestId": "OpenAPI",
        "serviceRequestVersion": "1.0",
        "channelId": _get_config_value("channel_id", "make_payment"),
    }

    encrypted_request = {
        "TransferPaymentRequest": {
            "SubHeader": sub_header,
            "TransferPaymentRequestBodyEncrypted": aes128_encrypt(
                _get_config_value("aes_symmetric_key"), json.dumps(request_body)
            ),
        }
    }

    plaintext_request = {
        "TransferPaymentRequest": {
            "SubHeader": sub_header,
            "TransferPaymentRequestBody": request_body,
        }
    }

    return Request(
        request_id,
        _get_config_value("url", "make_payment"),
        "make_payment",
        encrypted_request,
        plaintext_request,
    )


async def make_payment(axis_payment_request: AxisPaymentRequest) -> PaymentResponse:
    request = _make_payment_request(axis_payment_request)
    response = await _call(request)
    decrypted_body = aes128_decrypt(
        _get_config_value("aes_symmetric_key"),
        response["TransferPaymentResponse"]["TransferPaymentResponseBodyEncrypted"],
    )
    json_body = json.loads(decrypted_body)
    _log_response_body(request, json_body)
    if json_body["status"] != "S":
        raise PaymentProcessingError("Failed to make payment")
    return PaymentResponse(
        payment_transaction_id=axis_payment_request.internal_payment_transaction_id,
        transaction_status=PaymentTransactionStatusEnum.SUCCESS,
    )
