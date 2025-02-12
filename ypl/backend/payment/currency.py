from ypl.backend.config import settings
from ypl.db.payments import CurrencyEnum


def get_supported_currencies(country_code: str) -> list[CurrencyEnum]:
    all_supported_currencies = [
        CurrencyEnum.INR,
        CurrencyEnum.USD,
        CurrencyEnum.USDC,
        CurrencyEnum.ETH,
        CurrencyEnum.CBBTC,
        CurrencyEnum.BTC,
    ]
    # Only for testing
    if settings.ENVIRONMENT != "production":
        return all_supported_currencies

    if country_code == "IN":
        return [CurrencyEnum.INR]

    # We may filter some currencies based on country code in the future.
    return all_supported_currencies
