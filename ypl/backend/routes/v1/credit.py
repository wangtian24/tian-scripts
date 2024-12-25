import logging
from dataclasses import dataclass
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.exc import NoResultFound

from ypl.backend.llm.credit import (
    get_user_credit_balance,
)
from ypl.backend.payment.exchange_rates import get_exchange_rate
from ypl.backend.payment.facilitator import Facilitator, PaymentResponse
from ypl.backend.payment.validation import validate_destination_identifier_for_currency
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import (
    CurrencyEnum,
    PaymentInstrumentFacilitatorEnum,
    PaymentInstrumentIdentifierTypeEnum,
    PaymentTransactionStatusEnum,
)
from ypl.db.users import SIGNUP_CREDITS
from ypl.settings import settings

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
    # If True, return the response object instead of just the string.
    return_response_as_object: bool = False


async def convert_credits_to_currency(credits: int, currency: CurrencyEnum) -> Decimal:
    # TODO: Put them in a config somewhere.
    CREDITS_TO_INR_RATE = Decimal(0.1)
    CREDITS_TO_USD_RATE = Decimal(0.0012)

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

    if (
        (request.country_code is None or request.country_code == "IN")
        and settings.ENVIRONMENT == "production"
        and request.cashout_currency.is_crypto()
    ):
        raise HTTPException(status_code=400, detail="Crypto cashout is not supported in India")

    user_credit_balance = await get_user_credit_balance(request.user_id)
    if request.credits_to_cashout > user_credit_balance - SIGNUP_CREDITS:
        log_dict = {
            "message": "User does not have enough credits",
            "user_id": request.user_id,
            "credits_to_cashout": request.credits_to_cashout,
            "user_credit_balance": user_credit_balance,
        }
        logging.info(log_dict)
        raise HTTPException(status_code=400, detail="User does not have enough credits")

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


@router.post("/credits/cashout")
async def cashout_credits(request: CashoutCreditsRequest) -> str | None | PaymentResponse:
    await validate_cashout_request(request)

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

    facilitator = Facilitator.init(request.cashout_currency, request.destination_identifier_type, request.facilitator)
    payment_response = await facilitator.make_payment(
        request.user_id,
        request.credits_to_cashout,
        amount_in_currency,
        request.destination_identifier,
        request.destination_identifier_type,
    )

    # Hack to continue returning 'None' when the customer reference ID is not available,
    # as the older client will expect it.
    return payment_response if request.return_response_as_object else str(payment_response.customer_reference_id)


@router.get("/credits/cashout/transaction/{transaction_reference_id}/status")
async def get_cashout_transaction_status(transaction_reference_id: str) -> PaymentTransactionStatusEnum:
    facilitator = await Facilitator.for_transaction_reference_id(transaction_reference_id)
    return await facilitator.get_payment_status(transaction_reference_id)
