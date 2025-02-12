import asyncio
import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.exc import NoResultFound

from ypl.backend.config import settings
from ypl.backend.llm.credit import (
    Amount,
    CashoutInstrument,
    CashoutPaymentTransaction,
    CashoutUserInfo,
    RewardedCreditsPerActionTotals,
    get_cashout_amounts_per_currency,
    get_cashout_credit_stats,
    get_cashout_transactions,
    get_last_cashout_status,
    get_rewarded_credits_per_action_totals,
    get_total_credits_rank,
    get_total_credits_spent,
    get_user_credit_balance,
)
from ypl.backend.llm.utils import post_to_slack_with_user_name
from ypl.backend.payment.base_types import BaseFacilitator, PaymentResponse
from ypl.backend.payment.cashout_rate_limits import (
    CashoutKillswitchError,
    CashoutLimitError,
    check_cashout_killswitch,
    check_facilitator_cashout_killswitch,
    check_global_cashout_killswitch,
    check_request_rate_limit,
    log_cashout_limit_error,
    validate_and_return_cashout_user_limits,
)
from ypl.backend.payment.currency import get_supported_currencies
from ypl.backend.payment.exchange_rates import get_exchange_rate
from ypl.backend.payment.facilitator import get_supported_facilitators
from ypl.backend.payment.payment import get_last_successful_transaction_and_instrument, get_user_payment_instruments
from ypl.backend.payment.validation import validate_destination_identifier_for_currency
from ypl.backend.user.user import get_user
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.vpn_utils import get_ip_details
from ypl.db.payments import (
    CurrencyEnum,
    PaymentInstrumentFacilitatorEnum,
    PaymentInstrumentIdentifierTypeEnum,
)

router = APIRouter()

CREDITS_TO_INR_RATE = Decimal("0.05")
CREDITS_TO_USD_RATE = Decimal("0.002")
USD_TO_INR_RATE = Decimal("85")  # TODO: Get this from the exchange rate API. Hardcoded to match superset

SLACK_WEBHOOK_CASHOUT = settings.SLACK_WEBHOOK_CASHOUT


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
    ip_address: str | None = None


@dataclass
class CurrencyConversionResult:
    currency_amount: Decimal
    usd_amount: Decimal
    currency: CurrencyEnum


async def convert_credits_to_currency(credits: int, currency: CurrencyEnum) -> CurrencyConversionResult:
    # TODO: Put them in a config somewhere.

    credits_decimal: Decimal = Decimal(credits)

    log_dict = {
        "message": "Converting credits to currency",
        "credits_decimal": str(credits_decimal),
        "currency": currency.value,
        "currency_is_crypto": currency.is_crypto(),
    }
    logging.info(json_dumps(log_dict))

    usd_amount = credits_decimal * CREDITS_TO_USD_RATE

    if currency == CurrencyEnum.INR:
        inr_amount = credits_decimal * CREDITS_TO_INR_RATE
        exchange_rate = await get_exchange_rate(CurrencyEnum.USD, CurrencyEnum.INR)
        usd_amount = inr_amount / exchange_rate
        return CurrencyConversionResult(currency_amount=inr_amount, usd_amount=usd_amount, currency=currency)
    elif currency == CurrencyEnum.USD:
        return CurrencyConversionResult(currency_amount=usd_amount, usd_amount=usd_amount, currency=currency)
    elif currency == CurrencyEnum.USDC:
        return CurrencyConversionResult(currency_amount=usd_amount, usd_amount=usd_amount, currency=currency)
    else:
        exchange_rate = await get_exchange_rate(CurrencyEnum.USD, currency)
        return CurrencyConversionResult(
            currency_amount=usd_amount * exchange_rate, usd_amount=usd_amount, currency=currency
        )


async def validate_cashout_request(request: CashoutCreditsRequest) -> None:
    if request.credits_to_cashout <= 0:
        raise HTTPException(status_code=400, detail="You need to select at least a few credits to cash out.")

    try:
        await check_cashout_killswitch(request.facilitator, request.user_id)
    except CashoutKillswitchError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    user = await get_user(request.user_id)

    if (
        (request.country_code is None or request.country_code == "IN" or user.country_code == "IN")
        and settings.ENVIRONMENT == "production"
        and request.cashout_currency.is_crypto()
    ):
        log_dict = {
            "message": "Crypto cashout is not supported in India",
            "user_id": request.user_id,
            "credits_to_cashout": request.credits_to_cashout,
            "cashout_currency": request.cashout_currency,
            "passed_country_code": request.country_code,
            "user_country_code": user.country_code,
        }
        logging.warning(json_dumps(log_dict))

    if request.cashout_currency.is_crypto() and request.ip_address and settings.ENVIRONMENT == "production":
        ip_details = await get_ip_details(request.ip_address)
        if ip_details and ip_details["security"]["vpn"]:
            log_dict = {
                "message": "VPN detected. Blocking cashout to crypto.",
                "user_id": request.user_id,
                "ip_address": request.ip_address,
                "ip_details": ip_details,
                "credits_to_cashout": request.credits_to_cashout,
                "cashout_currency": request.cashout_currency,
                "passed_country_code": request.country_code,
                "user_country_code": user.country_code,
            }
            logging.warning(json_dumps(log_dict))
            asyncio.create_task(
                post_to_slack_with_user_name(request.user_id, json_dumps(log_dict), SLACK_WEBHOOK_CASHOUT)
            )
            raise HTTPException(status_code=400, detail="Cashout to crypto is not supported from a VPN")

    if (
        request.cashout_currency == CurrencyEnum.USD
        and request.facilitator == PaymentInstrumentFacilitatorEnum.CHECKOUT_COM
    ):
        if request.destination_additional_details is None:
            raise HTTPException(status_code=400, detail="Please enter your bank account details!")

    if request.cashout_currency == CurrencyEnum.USD and request.facilitator == PaymentInstrumentFacilitatorEnum.PLAID:
        if request.destination_additional_details is None:
            raise HTTPException(status_code=400, detail="Please enter your bank account details!")

        required_fields = ["account_number", "routing_number", "account_type", "user_name"]
        missing_fields = [field for field in required_fields if field not in request.destination_additional_details]
        if missing_fields:
            raise HTTPException(
                status_code=400, detail=f"Please enter the following details! {', '.join(missing_fields)}"
            )

    if (
        request.cashout_currency == CurrencyEnum.USD
        and request.facilitator == PaymentInstrumentFacilitatorEnum.HYPERWALLET
        and (
            request.destination_identifier_type == PaymentInstrumentIdentifierTypeEnum.PAYPAL_ID
            or request.destination_identifier_type == PaymentInstrumentIdentifierTypeEnum.VENMO_ID
        )
    ):
        if request.destination_additional_details is None:
            raise HTTPException(status_code=400, detail="Please enter additional details!")

    try:
        validate_destination_identifier_for_currency(request.cashout_currency, request.destination_identifier_type)
    except ValueError as e:
        log_dict = {
            "message": "Invalid destination identifier for currency",
            "user_id": request.user_id,
            "currency": request.cashout_currency,
            "identifier_type": request.destination_identifier_type,
        }
        logging.info(json_dumps(log_dict))
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
        "ip_address": request.ip_address,
    }
    logging.info(json_dumps(log_dict))

    await validate_cashout_request(request)
    try:
        await check_request_rate_limit(request.user_id)
        await validate_and_return_cashout_user_limits(request.user_id, request.credits_to_cashout)
    except CashoutLimitError as e:
        log_cashout_limit_error(e, is_precheck=False)
        raise HTTPException(status_code=400, detail=e.detail) from e
    except Exception as e:
        log_dict = {
            "message": "Error validating user cashout limits",
            "error": str(e),
            "user_id": request.user_id,
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail="Oops! Something went wrong. Please try again later.") from e

    try:
        conversion_result = await convert_credits_to_currency(request.credits_to_cashout, request.cashout_currency)
    except Exception as e:
        log_dict = {
            "message": "Error converting credits to currency",
            "error": str(e),
            "user_id": request.user_id,
            "credits_to_cashout": request.credits_to_cashout,
            "cashout_currency": request.cashout_currency,
        }
        logging.exception(json_dumps(log_dict))
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
            conversion_result.currency_amount,
            conversion_result.usd_amount,
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


@dataclass
class CashoutOptionsRequest:
    user_id: str
    guessed_country_code: str


@dataclass
class CashoutOptionsResponse:
    user_info: CashoutUserInfo

    supported_currencies: list[CurrencyEnum]
    supported_facilitators: list[PaymentInstrumentFacilitatorEnum]


@dataclass
class CashoutNotAvailableResponse:
    unavailable_reason: str


@router.post("/credits/cashout/options")
async def fetch_cashout_options(request: CashoutOptionsRequest) -> CashoutOptionsResponse | CashoutNotAvailableResponse:
    try:
        await check_global_cashout_killswitch(request.user_id)
    except CashoutKillswitchError as e:
        return CashoutNotAvailableResponse(unavailable_reason=str(e))

    [user, instruments, last_transaction_and_instrument] = await asyncio.gather(
        get_user(request.user_id),
        get_user_payment_instruments(request.user_id),
        get_last_successful_transaction_and_instrument(request.user_id),
    )

    user_country_code = user.country_code
    country_code = user_country_code if user_country_code else request.guessed_country_code

    facilitators = get_supported_facilitators(country_code)

    facilitator_killswitch_errors = await asyncio.gather(
        *[
            check_facilitator_cashout_killswitch(facilitator, request.user_id)  # noqa: F821
            for facilitator in facilitators
        ],
        return_exceptions=True,
    )

    filtered_facilitators = [
        facilitator
        for facilitator, error in zip(facilitators, facilitator_killswitch_errors, strict=True)
        if error is None
    ]

    currencies = get_supported_currencies(country_code)

    log_dict: dict[str, Any] = {}

    if not currencies or not filtered_facilitators:
        log_dict = {
            "message": "Cashout is not available",
            "user_id": request.user_id,
            "country_code": country_code,
            "user_country_code": user_country_code,
            "guessed_country_code": request.guessed_country_code,
            "currencies": [currency.value for currency in currencies],
            "facilitators": [facilitator.value for facilitator in facilitators],
            "facilitator_killswitch_errors": [str(error) for error in facilitator_killswitch_errors],
        }
        # Log this as an error because this is unexpected.
        logging.warning(json_dumps(log_dict))
        return CashoutNotAvailableResponse(unavailable_reason="Cashout is not available")

    try:
        user_limits = await validate_and_return_cashout_user_limits(request.user_id)
        user_info = CashoutUserInfo(
            credits_balance=user_limits.credits_balance,
            credits_available_for_cashout=user_limits.credits_available_for_cashout,
            minimum_credits_per_cashout=user_limits.minimum_credits_per_cashout,
            country_code=country_code,
            stored_instruments=[CashoutInstrument.from_payment_instrument(instrument) for instrument in instruments],
            last_successful_transaction=CashoutPaymentTransaction.from_payment_transaction(
                last_transaction_and_instrument[0],
                last_transaction_and_instrument[1],
                last_transaction_and_instrument[2],
            )
            if last_transaction_and_instrument
            else None,
        )

        log_dict = {
            "message": "Cashout options returned",
            "user_id": request.user_id,
            "user_info": user_info,
            "country_code": country_code,
            "user_country_code": user_country_code,
            "guessed_country_code": request.guessed_country_code,
            "currencies": [currency.value for currency in currencies],
            "facilitators": [facilitator.value for facilitator in facilitators],
        }
        logging.info(json_dumps(log_dict))
    except CashoutLimitError as e:
        log_cashout_limit_error(e, is_precheck=True)
        return CashoutNotAvailableResponse(unavailable_reason=e.detail)
    return CashoutOptionsResponse(
        user_info=user_info,
        supported_currencies=list(currencies),
        supported_facilitators=list(facilitators),
    )


@dataclass
class SummaryData:
    total_credits_rank: int
    rewarded_credits_per_action_totals: list[RewardedCreditsPerActionTotals]
    total_credits_won: int
    total_credits_spent: int
    last_cashout_status: CashoutPaymentTransaction | None
    number_of_cashout_transactions: int
    total_credits_cashed_out: int
    total_cashout_amounts_per_currency: list[Amount]


@dataclass
class CreditHistoryResponse:
    summary_data: SummaryData
    # Not adding server side pagination yet because we don't need it right now
    # due to low volume of rows in cashout transactions.
    cashout_transactions: list[CashoutPaymentTransaction]


async def generate_summary_data(user_id: str) -> SummaryData:
    [
        total_credits_rank,
        rewarded_credits_per_action_totals,
        total_credits_spent,
        last_cashout_status,
        cashout_credit_stats,
        cashout_amounts_per_currency,
    ] = await asyncio.gather(
        get_total_credits_rank(user_id),
        get_rewarded_credits_per_action_totals(user_id),
        get_total_credits_spent(user_id),
        get_last_cashout_status(user_id),
        get_cashout_credit_stats(user_id),
        get_cashout_amounts_per_currency(user_id),
    )
    total_credits_won = sum(rc.total_credits for rc in rewarded_credits_per_action_totals)
    number_of_cashout_transactions, total_credits_cashed_out = cashout_credit_stats
    return SummaryData(
        total_credits_rank=total_credits_rank,
        rewarded_credits_per_action_totals=rewarded_credits_per_action_totals,
        total_credits_won=total_credits_won,
        total_credits_spent=total_credits_spent,
        last_cashout_status=last_cashout_status,
        number_of_cashout_transactions=number_of_cashout_transactions,
        total_credits_cashed_out=total_credits_cashed_out,
        total_cashout_amounts_per_currency=cashout_amounts_per_currency,
    )


@router.get("/credits/history")
async def get_credit_history(user_id: str) -> CreditHistoryResponse:
    [summary_data, cashout_transactions] = await asyncio.gather(
        generate_summary_data(user_id),
        get_cashout_transactions(user_id),
    )
    return CreditHistoryResponse(summary_data=summary_data, cashout_transactions=cashout_transactions)
