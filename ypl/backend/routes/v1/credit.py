import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.exc import NoResultFound

from ypl.backend.config import settings
from ypl.backend.llm.credit import (
    get_user_credit_balance,
)
from ypl.backend.payment.base_types import BaseFacilitator, PaymentResponse
from ypl.backend.payment.cashout_rate_limits import (
    CashoutKillswitchError,
    check_cashout_killswitch,
    validate_user_cashout_limits,
)
from ypl.backend.payment.exchange_rates import get_exchange_rate
from ypl.backend.payment.validation import validate_destination_identifier_for_currency
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import (
    CurrencyEnum,
    PaymentInstrumentFacilitatorEnum,
    PaymentInstrumentIdentifierTypeEnum,
)
from ypl.db.users import SIGNUP_CREDITS

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
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@dataclass
class CashoutCreditsRequest:
    user_id: str
    credits_to_cashout: int
    cashout_currency: CurrencyEnum
    destination_identifier: str
    destination_identifier_type: PaymentInstrumentIdentifierTypeEnum
    facilitator: PaymentInstrumentFacilitatorEnum
    country_code: str | None
    # Optional dictionary for additional details like account number, routing number for USD payouts
    destination_additional_details: dict | None = None
    # Not all facilitators will need the validated destination details.
    validated_destination_details: str | None = None


async def convert_credits_to_currency(credits: int, currency: CurrencyEnum) -> Decimal:
    # TODO: Put them in a config somewhere.
    CREDITS_TO_INR_RATE = Decimal("0.05")
    CREDITS_TO_USD_RATE = Decimal("0.002")

    credits_decimal: Decimal = Decimal(credits)

    log_dict = {
        "message": "Converting credits to currency",
        "credits_decimal": str(credits_decimal),
        "currency": currency.value,
        "currency_is_crypto": currency.is_crypto(),
    }
    logging.info(log_dict)

    if currency == CurrencyEnum.INR:
        return credits_decimal * CREDITS_TO_INR_RATE
    elif currency == CurrencyEnum.USD:
        return credits_decimal * CREDITS_TO_USD_RATE
    elif currency == CurrencyEnum.USDC:
        return credits_decimal * CREDITS_TO_USD_RATE
    else:
        exchange_rate = await get_exchange_rate(CurrencyEnum.USD, currency)
        return credits_decimal * CREDITS_TO_USD_RATE * exchange_rate


async def validate_cashout_request(request: CashoutCreditsRequest) -> None:
    if request.credits_to_cashout <= 0:
        raise HTTPException(status_code=400, detail="You need to select at least a few credits to cash out.")

    try:
        await check_cashout_killswitch(request.facilitator, request.user_id)
    except CashoutKillswitchError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if (
        (request.country_code is None or request.country_code == "IN")
        and settings.ENVIRONMENT == "production"
        and request.cashout_currency.is_crypto()
    ):
        log_dict = {
            "message": "Crypto cashout is not supported in India",
            "user_id": request.user_id,
            "credits_to_cashout": request.credits_to_cashout,
            "cashout_currency": request.cashout_currency,
            "country_code": request.country_code,
        }
        logging.warning(log_dict)
        raise HTTPException(status_code=400, detail="Cashout to crypto is not supported in India")

    if request.cashout_currency == CurrencyEnum.USD and request.facilitator == PaymentInstrumentFacilitatorEnum.PLAID:
        if request.destination_additional_details is None:
            raise HTTPException(status_code=400, detail="Please enter your bank account details!")

        required_fields = ["account_number", "routing_number", "account_type", "user_name"]
        missing_fields = [field for field in required_fields if field not in request.destination_additional_details]
        if missing_fields:
            raise HTTPException(
                status_code=400, detail=f"Please enter the following details! {', '.join(missing_fields)}"
            )

    user_credit_balance = await get_user_credit_balance(request.user_id)
    if request.credits_to_cashout > user_credit_balance - SIGNUP_CREDITS:
        log_dict = {
            "message": "User does not have enough credits",
            "user_id": request.user_id,
            "credits_to_cashout": request.credits_to_cashout,
            "user_credit_balance": user_credit_balance,
        }
        logging.info(log_dict)
        raise HTTPException(status_code=400, detail="You do not have enough credits to cash out")

    try:
        validate_destination_identifier_for_currency(request.cashout_currency, request.destination_identifier_type)
    except ValueError as e:
        log_dict = {
            "message": "Invalid destination identifier for currency",
            "user_id": request.user_id,
            "currency": request.cashout_currency,
            "identifier_type": request.destination_identifier_type,
        }
        logging.info(log_dict)
        raise HTTPException(status_code=400, detail=str(e)) from e

    if request.validated_destination_details is None and request.facilitator == PaymentInstrumentFacilitatorEnum.UPI:
        raise HTTPException(status_code=400, detail="Please re-enter your UPI details!")


@router.post("/credits/cashout")
async def cashout_credits(request: CashoutCreditsRequest) -> str | None | PaymentResponse:
    # TODO: Remove this once things stabilize.
    log_dict = {
        "message": "Cashout request",
        "user_id": request.user_id,
        "credits_to_cashout": request.credits_to_cashout,
        "cashout_currency": request.cashout_currency,
        "destination_identifier": request.destination_identifier,
        "destination_identifier_type": request.destination_identifier_type,
        "facilitator": request.facilitator,
        "country_code": request.country_code,
        "destination_additional_details": request.destination_additional_details,
        "validated_destination_details_is_set": request.validated_destination_details is not None,
    }
    logging.info(json_dumps(log_dict))

    await validate_cashout_request(request)
    await validate_user_cashout_limits(request.user_id, request.credits_to_cashout)

    try:
        amount_in_currency = await convert_credits_to_currency(request.credits_to_cashout, request.cashout_currency)
    except Exception as e:
        log_dict = {
            "message": "Error converting credits to currency",
            "error": str(e),
            "user_id": request.user_id,
            "credits_to_cashout": request.credits_to_cashout,
            "cashout_currency": request.cashout_currency,
        }
        logging.exception(log_dict)
        raise HTTPException(
            status_code=500, detail=f"Error converting credits to currency {request.cashout_currency}"
        ) from e

    facilitator = BaseFacilitator.init(
        request.cashout_currency,
        request.destination_identifier_type,
        request.facilitator,
        request.destination_additional_details,
    )
    try:
        payment_response = await facilitator.make_payment(
            request.user_id,
            request.credits_to_cashout,
            amount_in_currency,
            request.destination_identifier,
            request.destination_identifier_type,
            request.destination_additional_details,
            request.validated_destination_details,
        )
    except ValueError as e:
        # Send all ValueError as 400 errors with the error message for the client to show.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException as e:
        # Let the HTTPException pass through as is. Client will show the error message on the UI for 4xx errors.
        raise e
    except Exception as e:
        # For all other exceptions, log the error and send a generic 500 error message to the client.
        log_dict = {
            "message": "Unexpected error during cashout",
            "error": str(e),
            "user_id": request.user_id,
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail="Oops! Something went wrong. Please try again later.") from e

    return payment_response


@router.get("/credits/cashout/transaction/{payment_transaction_id}/status")
async def get_cashout_transaction_status(payment_transaction_id: uuid.UUID) -> PaymentResponse:
    facilitator = await BaseFacilitator.for_payment_transaction_id(payment_transaction_id)
    return await facilitator.get_payment_status(payment_transaction_id)
