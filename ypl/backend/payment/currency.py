from ypl.db.payments import CurrencyEnum


def get_supported_currencies(country_code: str) -> list[CurrencyEnum]:
    if country_code == "IN":
        return [CurrencyEnum.INR]
    return [CurrencyEnum.USDC, CurrencyEnum.ETH]
