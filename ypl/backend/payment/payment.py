import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import DatabaseError, OperationalError
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from ypl.backend.db import get_async_session
from ypl.db.payments import (
    CurrencyEnum,
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
        transaction_data["created_at"] = datetime.utcnow()
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
