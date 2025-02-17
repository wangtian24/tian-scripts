from ypl.db.payments import CurrencyEnum, PaymentInstrumentIdentifierTypeEnum

supported_currency_to_instrument_identifier_type = {
    CurrencyEnum.INR: [PaymentInstrumentIdentifierTypeEnum.PHONE_NUMBER, PaymentInstrumentIdentifierTypeEnum.UPI_ID],
    CurrencyEnum.USD: [
        PaymentInstrumentIdentifierTypeEnum.BANK_ACCOUNT_NUMBER,
        PaymentInstrumentIdentifierTypeEnum.PARTNER_IDENTIFIER,
        PaymentInstrumentIdentifierTypeEnum.PAYPAL_ID,
        PaymentInstrumentIdentifierTypeEnum.VENMO_ID,
    ],
    CurrencyEnum.BTC: [
        PaymentInstrumentIdentifierTypeEnum.PHONE_NUMBER,
        PaymentInstrumentIdentifierTypeEnum.EMAIL,
    ],
    CurrencyEnum.CBBTC: [
        PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
    ],
    CurrencyEnum.DOGE: [
        PaymentInstrumentIdentifierTypeEnum.PHONE_NUMBER,
        PaymentInstrumentIdentifierTypeEnum.EMAIL,
    ],
    CurrencyEnum.ETH: [
        PaymentInstrumentIdentifierTypeEnum.PHONE_NUMBER,
        PaymentInstrumentIdentifierTypeEnum.EMAIL,
        PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
    ],
    CurrencyEnum.SOL: [
        PaymentInstrumentIdentifierTypeEnum.PHONE_NUMBER,
        PaymentInstrumentIdentifierTypeEnum.EMAIL,
    ],
    CurrencyEnum.USDC: [
        PaymentInstrumentIdentifierTypeEnum.PHONE_NUMBER,
        PaymentInstrumentIdentifierTypeEnum.EMAIL,
        PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
    ],
    CurrencyEnum.USDT: [
        PaymentInstrumentIdentifierTypeEnum.PHONE_NUMBER,
        PaymentInstrumentIdentifierTypeEnum.EMAIL,
    ],
}


def validate_destination_identifier_for_currency(
    currency: CurrencyEnum, identifier_type: PaymentInstrumentIdentifierTypeEnum
) -> None:
    if identifier_type not in supported_currency_to_instrument_identifier_type[currency]:
        raise ValueError(f"{currency} does not support transaction to a {identifier_type}")
