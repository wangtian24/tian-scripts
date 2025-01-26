import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import DatabaseError, OperationalError
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.payments import (
    CurrencyEnum,
    PaymentInstrument,
    PaymentTransaction,
    PaymentTransactionStatusEnum,
)
from ypl.db.point_transactions import PointsActionEnum, PointTransaction
from ypl.db.users import User


@dataclass
class PaymentTransactionRequest:
    currency: CurrencyEnum
    amount: Decimal
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
            select(PaymentTransaction).where(PaymentTransaction.payment_transaction_id == payment_transaction_id)  # type: ignore
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
        user_record = await session.execute(select(User).where(User.user_id == user_id))  # type: ignore
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
                .where(PointTransaction.user_id == user_id)  # type: ignore
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
                stmt = stmt.where(User.user_id == user_id)  # type: ignore

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
