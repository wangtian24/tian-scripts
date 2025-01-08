import logging
from dataclasses import dataclass
from decimal import Decimal

from fastapi import APIRouter, HTTPException

from ypl.backend.payment.base_types import (
    BaseFacilitator,
    PaymentDestinationIdentifierValidationError,
    ValidateDestinationIdentifierResponse,
)
from ypl.backend.payment.exchange_rates import get_exchange_rate
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
    logging.info(log_dict)

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
