import logging
from decimal import Decimal
from typing import Final

import httpx
from ypl.backend.config import settings
from ypl.db.payments import CurrencyEnum

CRYPTO_CURRENCY_IDS: Final[dict[CurrencyEnum, str]] = {
    CurrencyEnum.BTC: "bitcoin",
    CurrencyEnum.ETH: "ethereum",
    CurrencyEnum.USDC: "usd-coin",
    CurrencyEnum.CBBTC: "bitcoin",
}


async def get_crypto_exchange_rate(
    source_currency: CurrencyEnum,
    destination_currency: CurrencyEnum,
    client: httpx.AsyncClient,
) -> Decimal:
    """Get crypto exchange rate using primary (Coinbase) or fallback (CoinGecko) API."""
    log_dict = {
        "source_currency": source_currency.value,
        "destination_currency": destination_currency.value,
    }
    logging.info(log_dict)

    # Try primary API (Coinbase) first
    try:
        source_currency_id = source_currency.value.upper()

        if destination_currency == CurrencyEnum.CBBTC:
            destination_currency_id = "BTC"
        else:
            destination_currency_id = destination_currency.value.upper()

        response = await client.get(
            settings.CRYPTO_EXCHANGE_PRICE_API_URL_COINBASE.format(source_currency_id, destination_currency_id)
        )
        response.raise_for_status()
        data = response.json()
        exchange_rate = Decimal(str(data["data"]["amount"]))
        log_dict["exchange_rate_coinbase"] = str(exchange_rate)
        logging.info(log_dict)
        return exchange_rate

    except Exception as e:
        log_dict = {
            "message": "Coinbase API failed, trying CoinGecko",
            "source_currency": source_currency.value,
            "destination_currency": destination_currency.value,
            "error": str(e),
        }
        logging.warning(log_dict)
        # Only if Coinbase fails, try secondary API (CoinGecko)
        source_currency_id = source_currency.value.lower()
        destination_currency_id = CRYPTO_CURRENCY_IDS[destination_currency].lower()

        try:
            response = await client.get(
                settings.CRYPTO_EXCHANGE_PRICE_API_URL_COINGECKO.format(destination_currency_id, source_currency_id)
            )
            response.raise_for_status()
            data = response.json()

            # CoinGecko returns the exchange rate from destination currency to source currency.
            # We need to reverse it to get the exchange rate from source currency to destination currency.
            exchange_rate = 1 / Decimal(str(data[destination_currency_id][source_currency_id]))
            log_dict["exchange_rate_coingecko"] = str(exchange_rate)
            logging.info(log_dict)
            return exchange_rate

        except Exception as e:
            log_dict = {
                "message": "Failed to get crypto exchange rate from both APIs",
                "error": str(e),
                "source_currency": source_currency.value,
                "destination_currency": destination_currency.value,
            }
            logging.error(log_dict)
            raise ValueError(f"Failed to get crypto exchange rate from both APIs: {e}") from e


async def get_fiat_exchange_rate(
    source_currency: CurrencyEnum,
    destination_currency: CurrencyEnum,
    client: httpx.AsyncClient,
) -> Decimal:
    """Get fiat currency exchange rate.

    Args:
        source_currency: The source currency
        destination_currency: The destination currency
        client: The HTTP client to use for requests

    Returns:
        The exchange rate from source to destination currency
    """
    response = await client.get(
        f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/{source_currency.value.lower()}.json"
    )
    response.raise_for_status()
    data = response.json()
    return Decimal(data[source_currency.value.lower()][destination_currency.value.lower()])


async def get_exchange_rate(
    source_currency: CurrencyEnum,
    destination_currency: CurrencyEnum,
) -> Decimal:
    """Get exchange rate between two currencies.

    Args:
        source_currency: The source currency
        destination_currency: The destination currency

    Returns:
        The exchange rate from source to destination currency
    """
    if source_currency == destination_currency:
        return Decimal(1)

    log_dict = {
        "message": "Getting exchange rate",
        "source_currency": source_currency.value,
        "destination_currency": destination_currency.value,
        "is_crypto": source_currency.is_crypto() or destination_currency.is_crypto(),
    }
    logging.info(log_dict)

    async with httpx.AsyncClient() as client:
        if source_currency.is_crypto() or destination_currency.is_crypto():
            return await get_crypto_exchange_rate(source_currency, destination_currency, client)
        else:
            return await get_fiat_exchange_rate(source_currency, destination_currency, client)
