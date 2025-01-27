import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException

from ypl.backend.payment.base_types import (
    BaseFacilitator,
    PaymentDestinationIdentifierValidationError,
    ValidateDestinationIdentifierResponse,
)
from ypl.backend.payment.exchange_rates import get_exchange_rate
from ypl.backend.payment.payment import (
    PaymentInstrumentsResponse,
    UpdatePaymentInstrumentRequest,
    get_payment_instruments,
    update_payment_instrument,
)
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import CurrencyEnum, PaymentInstrumentFacilitatorEnum, PaymentInstrumentIdentifierTypeEnum

router = APIRouter()


@router.get("/payments/exchange-rate/{source_currency}/{destination_currency}")
async def exchange_rate(source_currency: CurrencyEnum, destination_currency: CurrencyEnum) -> Decimal:
    rate = await get_exchange_rate(source_currency, destination_currency)

    log_dict = {
        "message": "Exchange rate",
        "source_currency": source_currency.value,
        "destination_currency": destination_currency.value,
        "rate": rate,
    }
    logging.info(json_dumps(log_dict))

    return rate


@dataclass
class ValidateDestinationIdentifierRequest:
    destination_identifier: str
    destination_identifier_type: PaymentInstrumentIdentifierTypeEnum
    facilitator: PaymentInstrumentFacilitatorEnum
    currency: CurrencyEnum


@router.post("/payments/validate-destination-identifier")
async def validate_destination_identifier(
    request: ValidateDestinationIdentifierRequest,
) -> ValidateDestinationIdentifierResponse:
    facilitator = BaseFacilitator.init(
        request.currency,
        request.destination_identifier_type,
        request.facilitator,
    )
    try:
        return await facilitator.validate_destination_identifier(
            request.destination_identifier, request.destination_identifier_type
        )
    except PaymentDestinationIdentifierValidationError as e:
        log_dict = {
            "message": "Destination identifier validation failed at the facilitator",
            "error": str(e),
            "destination_identifier": request.destination_identifier,
            "destination_identifier_type": request.destination_identifier_type,
            "facilitator": request.facilitator,
            "currency": request.currency,
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log_dict = {
            "message": "Destination identifier validation failed due to an unexpected error",
            "error": str(e),
            "destination_identifier": request.destination_identifier,
            "destination_identifier_type": request.destination_identifier_type,
            "currency": request.currency,
            "facilitator": request.facilitator,
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail="Unexpected error, please try again later") from e


@router.get("/admin/payments/{user_id}/payment_instruments", response_model=PaymentInstrumentsResponse)
async def get_user_payment_instruments(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> PaymentInstrumentsResponse:
    """Get payment instruments for a user with pagination support."""
    log_dict: dict[str, Any] = {
        "message": "Getting payment instruments",
        "user_id": user_id,
        "limit": limit,
        "offset": offset,
    }
    logging.info(json_dumps(log_dict))
    try:
        response = await get_payment_instruments(user_id, limit, offset)

        log_dict = {
            "message": "Retrieved payment instruments",
            "user_id": user_id,
            "count": len(response.instruments),
            "has_more_rows": response.has_more_rows,
            "limit": limit,
            "offset": offset,
        }
        logging.info(json_dumps(log_dict))

        return response
    except Exception as e:
        log_dict = {
            "message": "Failed to retrieve payment instruments",
            "user_id": user_id,
            "error": str(e),
            "limit": limit,
            "offset": offset,
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail="Failed to retrieve payment instruments") from e


@router.patch("/admin/payments/payment_instruments/{payment_instrument_id}")
async def update_payment_instrument_endpoint(
    payment_instrument_id: UUID,
    request: UpdatePaymentInstrumentRequest,
) -> UUID:
    """Update a payment instrument's metadata and/or deleted_at status."""
    log_dict: dict[str, Any] = {
        "message": "Updating payment instrument",
        "payment_instrument_id": str(payment_instrument_id),
        "has_metadata_update": request.instrument_metadata is not None,
        "has_deleted_at_update": request.deleted_at is not None,
    }
    logging.info(json_dumps(log_dict))

    try:
        await update_payment_instrument(payment_instrument_id, request)

        log_dict = {
            "message": "Successfully updated payment instrument",
            "payment_instrument_id": str(payment_instrument_id),
        }
        logging.info(json_dumps(log_dict))

        return payment_instrument_id
    except Exception as e:
        log_dict = {
            "message": "Failed to update payment instrument",
            "payment_instrument_id": str(payment_instrument_id),
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail="Failed to update payment instrument") from e
