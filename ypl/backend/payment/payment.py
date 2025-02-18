import asyncio
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlmodel import func, select
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import (
    CurrencyEnum,
    DailyAccountBalanceHistory,
    PaymentInstrument,
    PaymentInstrumentFacilitatorEnum,
    PaymentInstrumentIdentifierTypeEnum,
    PaymentTransaction,
    PaymentTransactionStatusEnum,
)
from ypl.db.point_transactions import PointsActionEnum, PointTransaction
from ypl.db.users import User


@dataclass
class PaymentTransactionRequest:
    currency: CurrencyEnum
    amount: Decimal
    usd_amount: Decimal
    source_instrument_id: UUID
    destination_instrument_id: UUID
    status: PaymentTransactionStatusEnum
    additional_info: dict


@dataclass
class CashoutPointTransactionRequest:
    user_id: str
    point_delta: int
    action_type: PointsActionEnum
    cashout_payment_transaction_id: UUID
    action_details: dict | None = None


@dataclass
class PointTransactionResponse:
    transaction_id: str
    user_id: str
    point_delta: int
    action_type: PointsActionEnum
    action_details: dict
    created_at: datetime
    deleted_at: datetime | None


@dataclass
class PointTransactionsHistoryResponse:
    transactions: list[PointTransactionResponse]
    has_more_rows: bool


@dataclass
class CashoutResponse:
    created_at: datetime
    payment_transaction_id: UUID
    user_name: str
    email: str
    action_type: PointsActionEnum
    currency: CurrencyEnum
    amount: Decimal
    facilitator: str
    identifier: str
    instrument_metadata: dict
    customer_reference_id: str | None


@dataclass
class CashoutsHistoryResponse:
    cashouts: list[CashoutResponse]
    has_more_rows: bool


@dataclass
class PaymentInstrumentResponse:
    payment_instrument_id: UUID
    user_id: str
    facilitator: str
    identifier: str
    identifier_type: str
    metadata: dict
    created_at: datetime
    deleted_at: datetime | None


@dataclass
class PaymentInstrumentsResponse:
    instruments: list[PaymentInstrumentResponse]
    has_more_rows: bool


@dataclass
class UpdatePaymentInstrumentRequest:
    instrument_metadata: dict | None = None
    deleted_at: datetime | None = None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def create_payment_transaction(payment_transaction: PaymentTransactionRequest) -> UUID:
    """
    Create a new payment transaction with retry logic.

    Args:
        payment_transaction: The PaymentTransaction instance to create

    Returns:
        UUID: The payment transaction id
    """
    async with get_async_session() as session:
        transaction_data = asdict(payment_transaction)
        transaction_data["last_status_change_at"] = datetime.utcnow()
        payment_transaction_db = PaymentTransaction(**transaction_data)

        session.add(payment_transaction_db)
        await session.commit()
        return payment_transaction_db.payment_transaction_id


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def update_payment_transaction(payment_transaction_id: UUID, **fields_to_update: Any) -> None:
    """
    Update payment transaction fields with retry logic.

    Args:
        payment_transaction_id: The ID of the payment transaction to update
        **fields_to_update: Key-value pairs of fields to update
    """
    async with get_async_session() as session:
        payment_transaction_db = await session.execute(
            select(PaymentTransaction).where(PaymentTransaction.payment_transaction_id == payment_transaction_id)
        )
        payment_transaction_db_row = payment_transaction_db.scalar_one()

        for field, value in fields_to_update.items():
            setattr(payment_transaction_db_row, field, value)

        if "status" in fields_to_update:
            payment_transaction_db_row.last_status_change_at = datetime.utcnow()

        await session.commit()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def update_user_points(user_id: str, amount: int) -> None:
    async with get_async_session() as session:
        user_record = await session.execute(select(User).where(User.user_id == user_id))
        user_row = user_record.scalar_one()
        user_row.points += amount
        await session.commit()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def create_cashout_point_transaction(cashout_point_transaction_request: CashoutPointTransactionRequest) -> UUID:
    """
    Create a new point transaction for cashout with retry logic.

    Args:
        cashout_point_transaction_request: The CashoutPointTransactionRequest instance to create

    Returns:
        UUID: The point transaction id
    """
    async with get_async_session() as session:
        point_transaction_data = asdict(cashout_point_transaction_request)
        point_transaction_db = PointTransaction(**point_transaction_data)
        session.add(point_transaction_db)
        await session.commit()
        return point_transaction_db.transaction_id


async def get_points_transactions(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> PointTransactionsHistoryResponse:
    """Get points transaction history for a user with pagination support."""
    log_dict = {
        "message": "Getting points transactions",
        "user_id": user_id,
        "limit": limit,
        "offset": offset,
    }
    logging.info(json_dumps(log_dict))

    try:
        async with get_async_session() as session:
            stmt = (
                select(PointTransaction)
                .where(PointTransaction.user_id == user_id)
                .order_by(PointTransaction.created_at.desc())  # type: ignore
                .offset(offset)
                .limit(limit + 1)
            )

            result = await session.execute(stmt)
            transactions = result.scalars().all()

            has_more_rows = len(transactions) > limit
            if has_more_rows:
                transactions = transactions[:-1]

            log_dict = {
                "message": "Points transactions found",
                "user_id": user_id,
                "transactions_count": len(transactions),
                "limit": limit,
                "offset": offset,
                "has_more_rows": has_more_rows,
            }
            logging.info(json_dumps(log_dict))

            return PointTransactionsHistoryResponse(
                transactions=[
                    PointTransactionResponse(
                        transaction_id=str(tx.transaction_id),
                        user_id=tx.user_id,
                        point_delta=tx.point_delta,
                        action_type=tx.action_type,
                        action_details=tx.action_details,
                        created_at=tx.created_at,
                        deleted_at=tx.deleted_at,
                    )
                    for tx in transactions
                ],
                has_more_rows=has_more_rows,
            )

    except Exception as e:
        log_dict = {
            "message": "Error getting points transactions",
            "user_id": user_id,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from e


async def get_cashouts(
    limit: int = 50,
    offset: int = 0,
    user_id: str | None = None,
) -> CashoutsHistoryResponse:
    """Get cashout transaction history with pagination support."""
    log_dict = {
        "message": "Getting cashout transactions",
        "limit": limit,
        "offset": offset,
        "user_id": user_id,
    }
    logging.info(json_dumps(log_dict))

    try:
        async with get_async_session() as session:
            stmt = (
                select(PointTransaction, PaymentTransaction, PaymentInstrument, User)
                .select_from(PointTransaction)
                .join(
                    PaymentTransaction,
                    onclause=PointTransaction.cashout_payment_transaction_id
                    == PaymentTransaction.payment_transaction_id,  # type: ignore
                )
                .join(
                    PaymentInstrument,
                    onclause=PaymentTransaction.destination_instrument_id == PaymentInstrument.payment_instrument_id,  # type: ignore
                )
                .join(
                    User,
                    onclause=PaymentInstrument.user_id == User.user_id,  # type: ignore
                )
            )

            if user_id:
                stmt = stmt.where(User.user_id == user_id)

            stmt = stmt.order_by(PointTransaction.created_at.desc())  # type: ignore
            stmt = stmt.offset(offset)
            stmt = stmt.limit(limit + 1)

            result = await session.execute(stmt)
            rows = result.all()

            has_more_rows = len(rows) > limit
            if has_more_rows:
                rows = rows[:-1]

            log_dict = {
                "message": "Cashout transactions found",
                "transactions_count": len(rows),
                "limit": limit,
                "offset": offset,
                "has_more_rows": has_more_rows,
                "user_id": user_id,
            }
            logging.info(json_dumps(log_dict))

            return CashoutsHistoryResponse(
                cashouts=[
                    CashoutResponse(
                        payment_transaction_id=point_tx.cashout_payment_transaction_id,
                        created_at=point_tx.created_at,
                        action_type=point_tx.action_type,
                        currency=payment_tx.currency,
                        amount=payment_tx.amount,
                        customer_reference_id=payment_tx.customer_reference_id,
                        facilitator=str(payment_instrument.facilitator.value),
                        identifier=payment_instrument.identifier,
                        instrument_metadata=payment_instrument.instrument_metadata,
                        user_name=user.name,
                        email=user.email,
                    )
                    for point_tx, payment_tx, payment_instrument, user in rows
                ],
                has_more_rows=has_more_rows,
            )

    except Exception as e:
        log_dict = {
            "message": "Error getting cashout transactions",
            "error": str(e),
            "user_id": user_id,
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from e


async def validate_ledger_balance_all_users() -> None:
    """
    Validate user point balances against their transaction history.

    This function checks if each user's points balance matches the sum of their point_delta
    from the point_transactions table. If there's a mismatch, it logs a warning.
    """
    mismatched_users = 0
    log_dict: dict[str, Any] = {
        "message": "Validating ledger balance for all users",
    }
    logging.info(json_dumps(log_dict))

    async with get_async_session() as session:
        result = await session.execute(
            select(User).where(
                User.deleted_at.is_(None)  # type: ignore
            )
        )
        users = result.scalars().all()

        for user in users:
            point_delta_result = await session.execute(
                select(func.sum(PointTransaction.point_delta)).where(
                    PointTransaction.user_id == user.user_id,
                    PointTransaction.deleted_at.is_(None),  # type: ignore
                )
            )
            point_delta_sum = point_delta_result.scalar_one_or_none()

            point_delta_sum = point_delta_sum or 0

            if user.points != point_delta_sum:
                mismatched_users += 1
                log_dict = {
                    "message": "Mismatched ledger balance detected",
                    "user_id": user.user_id,
                    "user_name": user.name,
                    "user_points": user.points,
                    "points_sum": point_delta_sum,
                    "difference": user.points - point_delta_sum,
                }
                logging.warning(json_dumps(log_dict))

    if mismatched_users > 0:
        await post_to_slack(
            f":warning: Ledger balance validation found {mismatched_users} users with mismatched point balances."
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def get_payment_instruments(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> PaymentInstrumentsResponse:
    async with get_async_session() as session:
        query = (
            select(PaymentInstrument)
            .where(PaymentInstrument.user_id == user_id)
            .order_by(PaymentInstrument.created_at.desc())  # type: ignore
            .offset(offset)
            .limit(limit + 1)
        )

        result = await session.execute(query)
        instruments = result.scalars().all()

        has_more_rows = len(instruments) > limit
        if has_more_rows:
            instruments = instruments[:-1]

        log_dict = {
            "message": "Payment instruments found",
            "instruments_count": len(instruments),
            "limit": limit,
            "offset": offset,
            "has_more_rows": has_more_rows,
            "user_id": user_id,
        }
        logging.info(json_dumps(log_dict))

        return PaymentInstrumentsResponse(
            instruments=[
                PaymentInstrumentResponse(
                    payment_instrument_id=instrument.payment_instrument_id,
                    user_id=str(instrument.user_id),
                    facilitator=str(instrument.facilitator.value),
                    identifier=str(instrument.identifier),
                    identifier_type=str(instrument.identifier_type.value),
                    metadata=dict(instrument.instrument_metadata) if instrument.instrument_metadata else {},
                    created_at=instrument.created_at,
                    deleted_at=instrument.deleted_at,
                )
                for instrument in instruments
            ],
            has_more_rows=has_more_rows,
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def update_payment_instrument(payment_instrument_id: UUID, request: UpdatePaymentInstrumentRequest) -> None:
    """
    Update payment instrument fields with retry logic.

    Args:
        payment_instrument_id: The ID of the payment instrument to update
        request: The update request containing fields to update
    """
    async with get_async_session() as session:
        payment_instrument = await session.execute(
            select(PaymentInstrument).where(PaymentInstrument.payment_instrument_id == payment_instrument_id)
        )
        payment_instrument_row = payment_instrument.scalar_one()

        if payment_instrument_row is None:
            log_dict = {
                "message": "Payment instrument not found",
                "payment_instrument_id": str(payment_instrument_id),
            }
            logging.error(json_dumps(log_dict))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Payment instrument not found",
            )

        if request.instrument_metadata is not None:
            payment_instrument_row.instrument_metadata = request.instrument_metadata
        if request.deleted_at is not None:
            payment_instrument_row.deleted_at = request.deleted_at

        await session.commit()


async def retrieve_self_custodial_wallet_balances() -> list[DailyAccountBalanceHistory] | None:
    """
    Retrieve wallet balances from the daily_account_balances table.
    """
    try:
        async with get_async_session() as session:
            query = select(PaymentInstrument).where(
                func.coalesce(PaymentInstrument.user_id, "") == "SYSTEM",
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.ON_CHAIN,
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
                PaymentInstrument.deleted_at.is_(None),  # type: ignore
            )
            result = await session.execute(query)
            payment_instrument = result.scalar_one()

            # First find the latest date that has entries
            latest_date_query = select(func.date(func.max(DailyAccountBalanceHistory.created_at))).where(
                DailyAccountBalanceHistory.payment_instrument_id == payment_instrument.payment_instrument_id
            )
            result = await session.execute(latest_date_query)
            latest_date = result.scalar_one()

            if latest_date is None:
                return None

            query = (
                select(DailyAccountBalanceHistory)
                .where(
                    DailyAccountBalanceHistory.payment_instrument_id == payment_instrument.payment_instrument_id,  # type: ignore
                    func.date(DailyAccountBalanceHistory.created_at) == latest_date,
                )
                .order_by(DailyAccountBalanceHistory.created_at.desc())  # type: ignore
            )
            result = await session.execute(query)
            all_balances = list(result.scalars().all())

            # Get only the latest balance for each currency
            seen_currencies = set()
            balances = []
            for balance in all_balances:
                if balance.currency not in seen_currencies:
                    balances.append(balance)
                    seen_currencies.add(balance.currency)

            return balances
    except Exception as e:
        log_dict = {
            "message": "Failed to retrieve wallet balances",
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        return None


async def store_self_custodial_wallet_balances(wallet_data: dict) -> None:
    """
    Store wallet balances in the daily_account_balances table.

    Args:
        wallet_data: Dictionary containing wallet address and balances
            Format:
            {
                'wallet_address': str,
                'balances': [
                    {'currency': str, 'balance': Decimal},
                    ...
                ]
            }
    """
    try:
        async with get_async_session() as session:
            query = select(PaymentInstrument).where(
                func.coalesce(PaymentInstrument.user_id, "") == "SYSTEM",
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.ON_CHAIN,
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
                PaymentInstrument.deleted_at.is_(None),  # type: ignore
            )
            result = await session.execute(query)
            payment_instrument = result.scalar_one()

            # Create daily balance entries for each currency
            for balance_entry in wallet_data["balances"]:
                daily_balance = DailyAccountBalanceHistory(
                    payment_instrument_id=payment_instrument.payment_instrument_id,
                    account_id=wallet_data["wallet_address"],
                    currency=CurrencyEnum[balance_entry["currency"]],
                    balance=balance_entry["balance"],
                )
                session.add(daily_balance)

            await session.commit()

            log_dict = {
                "message": "Successfully stored wallet balances",
                "wallet_address": wallet_data["wallet_address"],
                "balances": [
                    {"currency": b["currency"], "balance": str(b["balance"])} for b in wallet_data["balances"]
                ],
            }
            logging.info(json_dumps(log_dict))

    except Exception as e:
        #  incase of any exception just post to slack. No need to raise an error
        #  as this is part of a daily cron job
        log_dict = {
            "message": "Failed to store wallet balances",
            "wallet_address": wallet_data.get("wallet_address"),
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        asyncio.create_task(post_to_slack(json_dumps(log_dict)))
        return None


async def retrieve_coinbase_retail_wallet_balances() -> list[DailyAccountBalanceHistory] | None:
    """
    Retrieve Coinbase retail wallet balances from the daily_account_balances table.
    """
    try:
        async with get_async_session() as session:
            query = select(PaymentInstrument).where(
                func.coalesce(PaymentInstrument.user_id, "") == "SYSTEM",
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.COINBASE,
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
                PaymentInstrument.deleted_at.is_(None),  # type: ignore
            )
            result = await session.execute(query)
            payment_instrument = result.scalar_one()

            # First find the latest date that has entries
            latest_date_query = select(func.date(func.max(DailyAccountBalanceHistory.created_at))).where(
                DailyAccountBalanceHistory.payment_instrument_id == payment_instrument.payment_instrument_id
            )
            result = await session.execute(latest_date_query)
            latest_date = result.scalar_one()

            if latest_date is None:
                return None

            query = (
                select(DailyAccountBalanceHistory)
                .where(
                    DailyAccountBalanceHistory.payment_instrument_id == payment_instrument.payment_instrument_id,  # type: ignore
                    func.date(DailyAccountBalanceHistory.created_at) == latest_date,
                )
                .order_by(DailyAccountBalanceHistory.created_at.desc())  # type: ignore
            )
            result = await session.execute(query)
            all_balances = list(result.scalars().all())

            # Get only the latest balance for each currency
            seen_currencies = set()
            balances = []
            for balance in all_balances:
                if balance.currency not in seen_currencies:
                    balances.append(balance)
                    seen_currencies.add(balance.currency)
            return balances
    except Exception as e:
        log_dict = {
            "message": "Failed to retrieve Coinbase retail wallet balances",
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        return None


async def store_coinbase_retail_wallet_balances(accounts: dict[str, dict[str, str | Decimal]]) -> None:
    """
    Store Coinbase retail wallet balances in the daily_account_balances table.
    Only stores balances for currencies that exist in CurrencyEnum.

    Args:
        accounts: Dictionary mapping currency codes to their account IDs and balances
            Format:
            {
                "currency_code": {
                    "account_id": str,
                    "balance": Decimal
                }
            }
    """
    try:
        async with get_async_session() as session:
            query = select(PaymentInstrument).where(
                func.coalesce(PaymentInstrument.user_id, "") == "SYSTEM",
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.COINBASE,
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
                PaymentInstrument.deleted_at.is_(None),  # type: ignore
            )
            result = await session.execute(query)
            payment_instrument = result.scalar_one()

            stored_balances = []
            skipped_currencies = []

            # Create daily balance entries for each currency
            for currency, details in accounts.items():
                try:
                    currency_enum = CurrencyEnum[currency]
                    daily_balance = DailyAccountBalanceHistory(
                        payment_instrument_id=payment_instrument.payment_instrument_id,
                        account_id=str(details["account_id"]),
                        currency=currency_enum,
                        balance=Decimal(str(details["balance"])),
                    )
                    session.add(daily_balance)
                    stored_balances.append({"currency": currency, "balance": str(details["balance"])})
                except KeyError:
                    skipped_currencies.append(currency)
                    continue

            await session.commit()

            log_dict = {
                "message": "Successfully stored Coinbase retail wallet balances",
                "balances": stored_balances,
            }
            if skipped_currencies:
                log_dict["skipped_currencies"] = skipped_currencies
                log_dict["reason"] = "Currencies not found in CurrencyEnum"
            logging.warning(json_dumps(log_dict))

    except Exception as e:
        #  incase of any exception just post to slack. No need to raise an error
        #  as this is part of a daily cron job
        log_dict = {
            "message": "Failed to store Coinbase retail wallet balances",
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        asyncio.create_task(post_to_slack(json_dumps(log_dict)))
        return None


async def retrieve_axis_upi_balance() -> Decimal | None:
    """
    Retrieve Axis UPI balance from the daily_account_balances table.
    """
    try:
        async with get_async_session() as session:
            query = select(PaymentInstrument).where(
                func.coalesce(PaymentInstrument.user_id, "") == "SYSTEM",
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.UPI,
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.UPI_ID,
                PaymentInstrument.deleted_at.is_(None),  # type: ignore
            )
            result = await session.execute(query)
            payment_instrument = result.scalar_one()

            # First find the latest date that has entries
            latest_date_query = select(func.date(func.max(DailyAccountBalanceHistory.created_at))).where(
                DailyAccountBalanceHistory.payment_instrument_id == payment_instrument.payment_instrument_id
            )
            result = await session.execute(latest_date_query)
            latest_date = result.scalar_one()

            if latest_date is None:
                return None

            query = (
                select(DailyAccountBalanceHistory)
                .where(
                    DailyAccountBalanceHistory.payment_instrument_id == payment_instrument.payment_instrument_id,  # type: ignore
                    func.date(DailyAccountBalanceHistory.created_at) == latest_date,
                )
                .order_by(DailyAccountBalanceHistory.created_at.desc())  # type: ignore
            )
            result = await session.execute(query)
            balance = result.scalar_one_or_none()
            if balance is None:
                return None

            return Decimal(str(balance.balance))
    except Exception as e:
        log_dict = {
            "message": "Failed to retrieve Axis UPI balance",
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        return None


async def store_axis_upi_balance(balance: Decimal) -> None:
    """
    Store Axis UPI balance in the daily_account_balances table.

    Args:
        balance: Current balance in INR from Axis UPI account
    """
    try:
        async with get_async_session() as session:
            query = select(PaymentInstrument).where(
                func.coalesce(PaymentInstrument.user_id, "") == "SYSTEM",
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.UPI,
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.UPI_ID,
                PaymentInstrument.deleted_at.is_(None),  # type: ignore
            )
            result = await session.execute(query)
            payment_instrument = result.scalar_one()

            daily_balance = DailyAccountBalanceHistory(
                payment_instrument_id=payment_instrument.payment_instrument_id,
                account_id=payment_instrument.identifier,
                currency=CurrencyEnum.INR,
                balance=balance,
            )
            session.add(daily_balance)

            await session.commit()

            log_dict = {
                "message": "Successfully stored Axis UPI balance",
                "balance": str(balance),
            }
            logging.info(json_dumps(log_dict))

    except Exception as e:
        # incase of any exception just post to slack. No need to raise an error
        # as this is part of a daily cron job
        log_dict = {
            "message": "Failed to store Axis UPI balance",
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        asyncio.create_task(post_to_slack(json_dumps(log_dict)))
        return None


# TODO: Merge with get_payment_instruments
async def get_user_payment_instruments(user_id: str) -> Sequence[PaymentInstrument]:
    async with get_async_session() as session:
        query = select(PaymentInstrument).where(
            PaymentInstrument.user_id == user_id,
            PaymentInstrument.deleted_at.is_(None),  # type: ignore
        )
        result = await session.exec(query)
        return result.all()


async def get_last_successful_transaction_and_instrument(
    user_id: str,
) -> tuple[PaymentTransaction, PaymentInstrument, PointTransaction] | None:
    async with get_async_session() as session:
        query = (
            select(PaymentTransaction, PaymentInstrument, PointTransaction)
            .join(
                PaymentInstrument,
                PaymentTransaction.destination_instrument_id == PaymentInstrument.payment_instrument_id,  # type: ignore
            )
            .join(
                PointTransaction,
                PointTransaction.cashout_payment_transaction_id == PaymentTransaction.payment_transaction_id,  # type: ignore
            )
            .where(
                PaymentInstrument.user_id == user_id,
                PaymentTransaction.status == PaymentTransactionStatusEnum.SUCCESS,
                PaymentTransaction.deleted_at.is_(None),  # type: ignore
            )
            .order_by(PaymentTransaction.created_at.desc())  # type: ignore
            .limit(1)
        )
        result = await session.exec(query)
        return result.one_or_none()


async def adjust_points(user_id: str, point_delta: int, reason: str) -> UUID:
    """Adjust points for a user."""
    async with get_async_session() as session:
        result = await session.exec(select(User).where(User.user_id == user_id))
        user = result.one_or_none()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        user.points += point_delta

        adjustment = PointTransaction(
            user_id=user.user_id,
            point_delta=point_delta,
            action_type=PointsActionEnum.ADJUSTMENT,
            action_details={"adjustment_reason": reason},
        )
        session.add(adjustment)

        await session.commit()

        return adjustment.transaction_id
