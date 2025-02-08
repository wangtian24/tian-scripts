import logging

from sqlalchemy import Connection, select
from sqlmodel import Session

from ypl.backend.payment.exchange_rates import get_exchange_rate
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import CurrencyEnum, PaymentTransaction


async def backfill_usd_amounts(connection: Connection) -> None:
    """Backfill USD amounts for all payment transactions based on their currency."""
    inr_exchange_rate = await get_exchange_rate(CurrencyEnum.USD, CurrencyEnum.INR)
    eth_exchange_rate = await get_exchange_rate(CurrencyEnum.USD, CurrencyEnum.ETH)
    with Session(connection) as session:
        stmt = select(PaymentTransaction)
        transactions = session.exec(stmt).scalars().all()  # type: ignore[call-overload]

        for transaction in transactions:
            try:
                if transaction.currency in [CurrencyEnum.USD, CurrencyEnum.USDC]:
                    transaction.usd_amount = transaction.amount
                else:
                    if transaction.currency == CurrencyEnum.INR:
                        exchange_rate = inr_exchange_rate
                    elif transaction.currency == CurrencyEnum.ETH:
                        exchange_rate = eth_exchange_rate
                    else:
                        log_dict = {
                            "message": "Unsupported currency",
                            "payment_transaction_id": str(transaction.payment_transaction_id),
                            "currency": transaction.currency.value,
                        }
                        logging.error(json_dumps(log_dict))
                        continue
                    transaction.usd_amount = transaction.amount / exchange_rate

                log_dict = {
                    "message": "Updated USD amount for payment transaction",
                    "payment_transaction_id": str(transaction.payment_transaction_id),
                    "currency": transaction.currency.value,
                    "currency_amount": str(transaction.amount),
                    "usd_amount": str(transaction.usd_amount),
                }
                logging.info(json_dumps(log_dict))

            except Exception as e:
                log_dict = {
                    "message": "Failed to update USD amount for payment transaction",
                    "payment_transaction_id": str(transaction.payment_transaction_id),
                    "currency": transaction.currency.value,
                    "error": str(e),
                }
                logging.error(json_dumps(log_dict))
                continue

        session.commit()
