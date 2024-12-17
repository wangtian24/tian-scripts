import logging
from decimal import Decimal

from fastapi import APIRouter

from ypl.backend.payment.exchange_rates import get_exchange_rate
from ypl.db.payments import CurrencyEnum

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
