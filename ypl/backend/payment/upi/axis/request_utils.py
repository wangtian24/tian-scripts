import base64
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Literal

import httpx
from ypl.backend.config import settings
from ypl.backend.payment.base_types import (
    PaymentDestinationIdentifierValidationError,
    PaymentProcessingError,
    PaymentResponse,
)
from ypl.backend.payment.upi.axis.cryptography_utils import (
    aes128_decrypt,
    aes128_encrypt,
    calculate_request_body_checksum,
)
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import PaymentInstrumentIdentifierTypeEnum, PaymentTransactionStatusEnum

RequestType = Literal["get_balance", "verify_vpa", "make_payment", "get_payment_status", "get_account_statement"]


@dataclass
class Request:
    request_id: str
    url: str
    request_type: RequestType
    encrypted_request: dict
    plaintext_request: dict
    internal_payment_transaction_id: uuid.UUID | None = None


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
        "payment_transaction_id": str(request.internal_payment_transaction_id),
        "timestamp": time.time(),
    }
    logging.info(json_dumps(log_dict))


def _log_response_body(request: Request, response_body: dict) -> None:
    log_dict = {
        "message": f"Decrypted response body from Axis {request.request_type} API",
        "request_id": request.request_id,
        "response_body": response_body,
        "payment_transaction_id": str(request.internal_payment_transaction_id),
        "timestamp": time.time(),
    }
    logging.info(json_dumps(log_dict))


# Axis requires a unique ID that's below 30 characters in alphanumeric.
# We plan to use the base32 encoded UUID, with the padding removed and converted to lowercase.
def _uuid_to_axis_unique_id(uuid: uuid.UUID) -> str:
    return base64.b32encode(uuid.bytes).decode("utf-8").replace("=", "").lower()


async def _call(request: Request) -> dict:
    _log_request(request)
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        response = await client.post(
            request.url,
            headers={
                "X-IBM-Client-Id": _get_config_value("client_id"),
                "X-IBM-Client-Secret": _get_config_value("client_secret"),
                "Referer": request.request_id,
            },
            json=request.encrypted_request,
        )

        response_json = None
        try:
            response_json = response.json()
            logging.info(
                json_dumps(
                    {
                        "message": f"Received response from Axis {request.request_type} API",
                        "status_code": response.status_code,
                        "response": response_json,
                        "request_id": request.request_id,
                        "payment_transaction_id": str(request.internal_payment_transaction_id),
                        "duration": time.time() - start_time,
                        "timestamp": time.time(),
                    }
                )
            )
            response.raise_for_status()
            return response_json  # type: ignore[no-any-return]
        except httpx.HTTPStatusError as e:
            logging.error(
                json_dumps(
                    {
                        "message": f"HTTP error in Axis {request.request_type} API",
                        "status_code": response.status_code,
                        "response": response_json,
                        "error": str(e),
                        "request_id": request.request_id,
                        "payment_transaction_id": str(request.internal_payment_transaction_id),
                        "duration": time.time() - start_time,
                        "timestamp": time.time(),
                    }
                )
            )
            raise


def _make_get_balance_request(internal_payment_transaction_id: uuid.UUID | None = None) -> Request:
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
        internal_payment_transaction_id,
    )


def _mock_staging_get_balance_response() -> dict:
    if settings.ENVIRONMENT == "production":
        raise Exception("Cannot mock response in production")
    return {
        "status": "S",
        "data": {
            "Balance": "1000.00",
        },
    }


async def get_balance(internal_payment_transaction_id: uuid.UUID | None = None) -> Decimal:
    request = _make_get_balance_request(internal_payment_transaction_id)
    try:
        response = await _call(request)
        decrypted_body = aes128_decrypt(
            _get_config_value("aes_symmetric_key"),
            response["GetAccountBalanceResponse"]["GetAccountBalanceResponseBodyEncrypted"],
        )
        json_body = json.loads(decrypted_body)
        _log_response_body(request, json_body)
        # Mock staging response for failures.
        if json_body["status"] != "S" and settings.ENVIRONMENT != "production":
            json_body = _mock_staging_get_balance_response()
            logging.info(
                json_dumps(
                    {
                        "message": "Mocked staging get balance response",
                        "request_id": request.request_id,
                        "modified_response_body": json_body,
                        "payment_transaction_id": str(request.internal_payment_transaction_id),
                    }
                )
            )
    except Exception as e:
        logging.error(f"Error getting balance: {e}")
        if settings.ENVIRONMENT != "production":
            json_body = _mock_staging_get_balance_response()
            logging.info(
                json_dumps(
                    {
                        "message": "Mocked staging get balance response",
                        "request_id": request.request_id,
                        "modified_response_body": json_body,
                        "payment_transaction_id": str(request.internal_payment_transaction_id),
                    }
                )
            )
        else:
            raise

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
    # VPA of the beneficiary. Phone number will not work.
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
                # It is never the phone number.
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
    # The transaction status is not started until the bank acknowledges the request.
    transaction_status = PaymentTransactionStatusEnum.NOT_STARTED
    try:
        response = await _call(request)
        decrypted_body = aes128_decrypt(
            _get_config_value("aes_symmetric_key"),
            response["TransferPaymentResponse"]["TransferPaymentResponseBodyEncrypted"],
        )
        json_body = json.loads(decrypted_body)
        _log_response_body(request, json_body)
        if json_body["status"] != "S":
            raise PaymentProcessingError("Partner rejected the payment request")
        # The bank acknowledges the request, we can set the transaction status to pending.
        transaction_status = PaymentTransactionStatusEnum.PENDING
    except PaymentProcessingError:
        # The partner rejected the payment request, let the caller handle it via the exception handler.
        raise
    except Exception as e:
        # We don't know the status of the error, so we let the status be NOT_STARTED,
        # and let the caller handle it via the retry loop that checks the status.
        log_dict = {
            "message": "Failed during the payment call to the partner",
            "error": str(e),
            "payment_transaction_id": str(axis_payment_request.internal_payment_transaction_id),
            "request_id": request.request_id,
            "transaction_status": transaction_status,  # will be NOT_STARTED
        }
        logging.exception(json_dumps(log_dict))

    return PaymentResponse(
        payment_transaction_id=axis_payment_request.internal_payment_transaction_id,
        transaction_status=transaction_status,
        partner_reference_id=_uuid_to_axis_unique_id(axis_payment_request.internal_payment_transaction_id),
    )


def _make_get_payment_status_request(
    partner_reference_id: str, internal_payment_transaction_id: uuid.UUID | None = None
) -> Request:
    request_id = str(uuid.uuid4())

    request_body = {
        "channelId": _get_config_value("channel_id", "get_payment_status"),
        "corpCode": _get_config_value("corp_code", "get_payment_status"),
        "crn": partner_reference_id,
    }
    request_body["checksum"] = calculate_request_body_checksum(request_body)

    sub_header = {
        "requestUUID": request_id,
        "serviceRequestId": "OpenAPI",
        "serviceRequestVersion": "1.0",
        "channelId": _get_config_value("channel_id", "get_payment_status"),
    }

    encrypted_request = {
        "GetStatusRequest": {
            "SubHeader": sub_header,
            "GetStatusRequestBodyEncrypted": aes128_encrypt(
                _get_config_value("aes_symmetric_key"), json.dumps(request_body)
            ),
        }
    }

    plaintext_request = {
        "GetStatusRequest": {
            "SubHeader": sub_header,
            "GetStatusRequestBody": request_body,
        }
    }

    return Request(
        request_id,
        _get_config_value("url", "get_payment_status"),
        "get_payment_status",
        encrypted_request,
        plaintext_request,
        internal_payment_transaction_id,
    )


def _map_transaction_status(transaction_status: str) -> PaymentTransactionStatusEnum | None:
    if transaction_status == "PENDING":
        return PaymentTransactionStatusEnum.PENDING
    if transaction_status == "PROCESSED":
        return PaymentTransactionStatusEnum.SUCCESS
    if transaction_status == "REJECTED":
        return PaymentTransactionStatusEnum.FAILED
    if transaction_status == "RETURNED":
        # The beneficiary account rejected the payment. Mark this one as failed..
        return PaymentTransactionStatusEnum.FAILED
    return None


def _mock_staging_get_payment_status_response(partner_reference_id: str) -> dict:
    if settings.ENVIRONMENT == "production":
        raise Exception("Cannot mock response in production")
    return {
        "status": "S",
        "data": {
            "CUR_TXN_ENQ": [
                {
                    "crn": partner_reference_id,
                    "transactionStatus": "PROCESSED",
                    # Mock a random UTR for the transaction based on the partner reference ID.
                    "utrNo": str(abs(hash(partner_reference_id)))[:10],
                }
            ],
        },
    }


async def get_payment_status(internal_payment_transaction_id: uuid.UUID, partner_reference_id: str) -> PaymentResponse:
    request = _make_get_payment_status_request(partner_reference_id, internal_payment_transaction_id)
    response = await _call(request)
    decrypted_body = aes128_decrypt(
        _get_config_value("aes_symmetric_key"),
        response["GetStatusResponse"]["GetStatusResponseBodyEncrypted"],
    )
    json_body = json.loads(decrypted_body)
    _log_response_body(request, json_body)

    # Mock staging response for failures 25% of the time, so that the user doesn't see an immediate response,
    # and it imitates the behavior of production.
    if settings.ENVIRONMENT != "production" and random.random() > 0.75:
        json_body = _mock_staging_get_payment_status_response(partner_reference_id)
        logging.info(
            json_dumps(
                {
                    "message": "Mocked staging get payment status response",
                    "request_id": request.request_id,
                    "modified_response_body": json_body,
                    "payment_transaction_id": str(internal_payment_transaction_id),
                }
            )
        )

    if json_body["status"] != "S":
        raise PaymentProcessingError("Failed to get payment status")

    status_items = json_body["data"]["CUR_TXN_ENQ"]
    # If there are no status items, the transaction may not have reached the partner (yet).
    # We'll set the status to NOT_STARTED, and let the caller handle the status via the retry loop.
    # If these cases happen, they may only be resolved via offline reconciliation.
    transaction_status: PaymentTransactionStatusEnum | None = (
        PaymentTransactionStatusEnum.PENDING if len(status_items) > 0 else PaymentTransactionStatusEnum.NOT_STARTED
    )
    customer_reference_id = None
    # Iterate in reverse to get the status of the last attempt for this reference ID.
    for status_item in reversed(status_items):
        if status_item["crn"] == partner_reference_id:
            transaction_status = _map_transaction_status(status_item["transactionStatus"])
            if transaction_status is None:
                transaction_status = PaymentTransactionStatusEnum.PENDING
                logging.error(
                    json_dumps(
                        {
                            "message": "Unknown transaction status from Axis UPI",
                            "partner_transaction_status": status_item["transactionStatus"],
                            "partner_reference_id": partner_reference_id,
                            "payment_transaction_id": str(internal_payment_transaction_id),
                        }
                    )
                )
            # utrNo will be None if the transaction is pending.
            customer_reference_id = status_item["utrNo"]
            break

    # We're handling the case where the transaction status is None in the loop above.
    assert transaction_status is not None

    return PaymentResponse(
        payment_transaction_id=internal_payment_transaction_id,
        transaction_status=transaction_status,
        customer_reference_id=customer_reference_id,
    )


def _make_verify_vpa_request(vpa: str) -> Request:
    request_id = str(uuid.uuid4())

    request_body = {
        "merchId": _get_config_value("merchant_id", "verify_vpa"),
        "merchChanId": _get_config_value("merchant_channel_id", "verify_vpa"),
        "customerVpa": vpa,
        "corpCode": _get_config_value("corp_code", "verify_vpa"),
        "channelId": _get_config_value("channel_id", "verify_vpa"),
    }
    request_body["checksum"] = calculate_request_body_checksum(request_body)

    sub_header = {
        "requestUUID": request_id,
        "serviceRequestId": "OpenAPI",
        "serviceRequestVersion": "1.0",
        "channelId": _get_config_value("channel_id", "verify_vpa"),
    }

    encrypted_request = {
        "VerifyVPARequest": {
            "SubHeader": sub_header,
            "VerifyVPARequestBodyEncrypted": aes128_encrypt(
                _get_config_value("aes_symmetric_key"), json_dumps(request_body)
            ),
        }
    }

    plaintext_request = {
        "VerifyVPARequest": {
            "SubHeader": sub_header,
        },
        "VerifyVPARequestBody": request_body,
    }

    return Request(
        request_id,
        _get_config_value("url", "verify_vpa"),
        "verify_vpa",
        encrypted_request,
        plaintext_request,
    )


@dataclass
class VerifyVpaResponse:
    validated_vpa: str
    customer_name: str


def _map_verify_vpa_error(error_code: str, destination_identifier_type: PaymentInstrumentIdentifierTypeEnum) -> str:
    is_phone_number = destination_identifier_type == PaymentInstrumentIdentifierTypeEnum.PHONE_NUMBER
    if error_code == "MM2":
        return (
            "This phone number is not linked to a valid UPI ID. Please check the number and try again."
            if is_phone_number
            else "Please check the UPI ID and try again"
        )
    if error_code == "MM3":
        return (
            "This phone number is not linked to a valid UPI ID. Please check the number and try again."
            if is_phone_number
            else "This UPI ID is blocked, please try with a different UPI ID"
        )

    if error_code == "MM4":
        return (
            "The associated UPI ID for this phone number is inactive"
            if is_phone_number
            else "This UPI ID is inactive, please try with a different UPI ID"
        )
    if error_code == "ZH":
        return (
            "This phone number is not linked to a valid UPI ID. Please check the number and try again."
            if is_phone_number
            else "This UPI ID is not valid, please try with a different UPI ID"
        )
    raise Exception("Unknown error")


def _mock_staging_verify_vpa_response(
    destination_identifier: str, destination_identifier_type: PaymentInstrumentIdentifierTypeEnum
) -> dict:
    if settings.ENVIRONMENT == "production":
        raise Exception("Cannot mock response in production")
    if destination_identifier == "1234567890":
        return {
            "result": "SUCCESS",
            "vpa": "test@okaxis",
            "customerName": "Test User",
        }
    if destination_identifier == "9876543210@okaxis":
        return {
            "result": "SUCCESS",
            "customerName": "Test User",
            "vpa": None,
        }
    if destination_identifier == "2345678901":
        return {
            "result": "MM2",
            "code": "MM2",
            "customerName": "",
            "vpa": None,
        }
    if destination_identifier == "3456789012":
        return {
            "result": "MM3",
            "code": "MM3",
            "customerName": "",
            "vpa": None,
        }
    if destination_identifier == "4567890123":
        return {
            "result": "MM4",
            "code": "MM4",
            "customerName": "",
            "vpa": None,
        }
    return {
        "result": "ZH",
        "code": "ZH",
        "customerName": "",
        "vpa": None,
    }


async def verify_vpa(
    destination_identifier: str, destination_identifier_type: PaymentInstrumentIdentifierTypeEnum
) -> VerifyVpaResponse:
    request = _make_verify_vpa_request(destination_identifier)
    try:
        response = await _call(request)
        decrypted_body = aes128_decrypt(
            _get_config_value("aes_symmetric_key"),
            response["VerifyVPAResponse"]["VerifyVPAResponseBodyEncrypted"],
        )
        json_body = json.loads(decrypted_body)
        _log_response_body(request, json_body)

        if json_body["result"] != "SUCCESS" and settings.ENVIRONMENT != "production":
            json_body = _mock_staging_verify_vpa_response(destination_identifier, destination_identifier_type)
            logging.info(
                json_dumps(
                    {
                        "message": "Mocked staging verify VPA response",
                        "request_id": request.request_id,
                        "modified_response_body": json_body,
                    }
                )
            )
    except Exception as e:
        logging.error(
            json_dumps(
                {
                    "message": "Error verifying VPA",
                    "error": str(e),
                    "request_id": request.request_id,
                }
            )
        )
        if settings.ENVIRONMENT != "production":
            json_body = _mock_staging_verify_vpa_response(destination_identifier, destination_identifier_type)
            logging.info(
                json_dumps(
                    {
                        "message": "Mocked staging verify VPA response",
                        "request_id": request.request_id,
                        "modified_response_body": json_body,
                    }
                )
            )
        else:
            raise

    if json_body["result"] != "SUCCESS":
        message = _map_verify_vpa_error(json_body["result"], destination_identifier_type)
        raise PaymentDestinationIdentifierValidationError(message)
    validated_vpa = destination_identifier
    if destination_identifier_type == PaymentInstrumentIdentifierTypeEnum.PHONE_NUMBER:
        validated_vpa = json_body["vpa"]
        if not isinstance(validated_vpa, str):
            raise PaymentDestinationIdentifierValidationError(
                f"Could not find a valid VPA for the given phone number: {destination_identifier}"
            )
    return VerifyVpaResponse(
        validated_vpa=validated_vpa,
        customer_name=json_body["customerName"],
    )
