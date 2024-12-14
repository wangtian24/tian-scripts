from decimal import Decimal

import httpx
from ypl.db.payments import CurrencyEnum


async def get_exchange_rate(source_currency: CurrencyEnum, destination_currency: CurrencyEnum) -> Decimal:
    if source_currency == destination_currency:
        return Decimal(1)

    async with httpx.AsyncClient() as client:
        # TODO: Find a better source of real-time exchange rates.
        response = await client.get(
            f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/{source_currency}.json"
        )
        response.raise_for_status()
        data = response.json()
        return Decimal(data[source_currency][destination_currency])
