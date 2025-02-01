import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy import Enum, ScalarResult, Select, func
from sqlalchemy.engine import TupleResult
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlmodel import select
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ypl.backend.db import get_async_session
from ypl.db.payments import (
    CurrencyEnum,
    PaymentInstrument,
    PaymentInstrumentFacilitatorEnum,
    PaymentInstrumentIdentifierTypeEnum,
    PaymentTransaction,
    PaymentTransactionStatusEnum,
)
from ypl.db.point_transactions import PointsActionEnum, PointTransaction
from ypl.db.rewards import Reward, RewardActionEnum, RewardActionLog, RewardStatusEnum
from ypl.db.users import User


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def get_user_credit_balance(user_id: str) -> int:
    query = select(User.points).where(
        User.user_id == user_id,
        User.deleted_at.is_(None),  # type: ignore
    )

    async with get_async_session() as session:
        result = await session.exec(query)
        return result.one()


@dataclass
class Amount:
    currency: CurrencyEnum
    amount: Decimal
    credits: int


@dataclass
class RewardedCreditsPerActionTotals:
    action_type: RewardActionEnum
    total_credits: int


@dataclass
class DateTimeValue:
    datetime: datetime
    value: int


@dataclass
class CashoutPaymentTransaction:
    payment_transaction_id: uuid.UUID
    created_at: datetime
    credits_cashed_out: int
    currency: CurrencyEnum
    amount: Decimal
    status: PaymentTransactionStatusEnum
    last_status_change_at: datetime
    facilitator: PaymentInstrumentFacilitatorEnum
    destination_identifier: str
    destination_identifier_type: PaymentInstrumentIdentifierTypeEnum
    customer_reference_id: str | None


async def get_total_credits_rank(user_id: str) -> int:
    async with get_async_session() as session:
        user_total: Select = select(func.coalesce(func.sum(PointTransaction.point_delta), 0).label("user_total")).where(
            PointTransaction.user_id == user_id, PointTransaction.action_type == PointsActionEnum.REWARD
        )

        rank_query = select(func.count().label("rank")).select_from(
            select(PointTransaction.user_id, func.sum(PointTransaction.point_delta).label("total"))
            .where(PointTransaction.action_type == PointsActionEnum.REWARD)
            .group_by(PointTransaction.user_id)  # type: ignore
            .having(func.sum(PointTransaction.point_delta) > user_total)
            .subquery()
        )

        result: ScalarResult[int] = await session.exec(rank_query)
        return result.one()


async def get_rewarded_credits_per_action_totals(user_id: str) -> list[RewardedCreditsPerActionTotals]:
    async with get_async_session() as session:
        result: TupleResult = await session.exec(
            select(
                func.sum(Reward.credit_delta).label("total_credits"),
                func.coalesce(
                    RewardActionLog.action_type,
                    func.cast(func.text(RewardActionEnum.MODEL_FEEDBACK.name), Enum(RewardActionEnum)),
                ).label("action_type"),
            )
            .outerjoin(RewardActionLog, RewardActionLog.associated_reward_id == Reward.reward_id)  # type: ignore
            .where(Reward.user_id == user_id, Reward.status == RewardStatusEnum.CLAIMED)
            .group_by("action_type")
        )
        return [
            RewardedCreditsPerActionTotals(action_type=action_type, total_credits=total_credits)
            for total_credits, action_type in result.all()
        ]


async def get_total_credits_spent(user_id: str) -> int:
    async with get_async_session() as session:
        result = await session.exec(
            select(func.abs(func.coalesce(func.sum(PointTransaction.point_delta), 0)).label("total_credits")).where(
                PointTransaction.user_id == user_id, PointTransaction.action_type == PointsActionEnum.PROMPT
            )
        )
        return result.one()  # type: ignore


def cashout_query(user_id: str) -> Select:
    return (
        select(PaymentTransaction, PaymentInstrument, PointTransaction)
        .join(
            PaymentTransaction,
            PaymentTransaction.payment_transaction_id == PointTransaction.cashout_payment_transaction_id,  # type: ignore
        )
        .join(
            PaymentInstrument,
            PaymentTransaction.destination_instrument_id == PaymentInstrument.payment_instrument_id,  # type: ignore
        )
        .where(PaymentInstrument.user_id == user_id)
    )


def make_cashout_payment_transaction(
    payment_transaction: PaymentTransaction,
    destination_payment_instrument: PaymentInstrument,
    point_transaction: PointTransaction,
) -> CashoutPaymentTransaction:
    assert payment_transaction.created_at is not None

    return CashoutPaymentTransaction(
        payment_transaction_id=payment_transaction.payment_transaction_id,
        created_at=payment_transaction.created_at,
        credits_cashed_out=(
            # invert the sign to show positive credits cashed out, and negative credits for reversed
            -1 * point_transaction.point_delta
        ),
        currency=payment_transaction.currency,
        amount=payment_transaction.amount,
        status=payment_transaction.status,
        last_status_change_at=payment_transaction.last_status_change_at,
        facilitator=destination_payment_instrument.facilitator,
        destination_identifier=destination_payment_instrument.identifier,
        destination_identifier_type=destination_payment_instrument.identifier_type,
        customer_reference_id=payment_transaction.customer_reference_id,
    )


async def get_last_cashout_status(user_id: str) -> CashoutPaymentTransaction | None:
    async with get_async_session() as session:
        result = await session.exec(
            cashout_query(user_id)
            .order_by(PaymentTransaction.modified_at.desc())  # type: ignore
            .limit(1)
        )
        row = result.one_or_none()
        if row is None:
            return None
        [payment_transaction, payment_instrument, point_transaction] = row
        assert payment_transaction.last_status_change_at is not None
        return make_cashout_payment_transaction(payment_transaction, payment_instrument, point_transaction)


async def get_cashout_amounts_per_currency(user_id: str) -> list[Amount]:
    async with get_async_session() as session:
        result = await session.exec(
            select(  # type: ignore
                PaymentTransaction.currency,
                func.sum(PaymentTransaction.amount).label("total_amount"),
                func.abs(func.sum(PointTransaction.point_delta)).label("total_credits"),
            )
            .join(
                PaymentInstrument,
                PaymentTransaction.destination_instrument_id == PaymentInstrument.payment_instrument_id,
            )
            .join(
                PointTransaction,
                PaymentTransaction.payment_transaction_id == PointTransaction.cashout_payment_transaction_id,
            )
            .where(
                PaymentInstrument.user_id == user_id,
                PaymentTransaction.status == PaymentTransactionStatusEnum.SUCCESS,
            )
            .group_by(PaymentTransaction.currency)
        )
        return [
            Amount(currency=currency, amount=total_amount, credits=total_credits)
            for currency, total_amount, total_credits in result.all()
        ]


async def get_cashout_credit_stats(user_id: str) -> tuple[int, int]:
    async with get_async_session() as session:
        result = await session.exec(
            select(
                func.coalesce(func.count(PaymentTransaction.payment_transaction_id), 0).label(  # type: ignore
                    "number_of_cashout_transactions"
                ),
                func.coalesce(func.abs(func.sum(PointTransaction.point_delta)), 0).label("total_credits_cashed_out"),
            )
            .join(
                PaymentTransaction,
                PaymentTransaction.payment_transaction_id == PointTransaction.cashout_payment_transaction_id,  # type: ignore
            )
            .where(
                PointTransaction.user_id == user_id,
                PointTransaction.action_type == PointsActionEnum.CASHOUT,
                PaymentTransaction.status == PaymentTransactionStatusEnum.SUCCESS,
            )
        )
        row = result.one_or_none()
        if row is None:
            return 0, 0
        [number_of_cashout_transactions, total_credits_cashed_out] = row
        return number_of_cashout_transactions, total_credits_cashed_out


async def get_cashout_transactions(user_id: str) -> list[CashoutPaymentTransaction]:
    async with get_async_session() as session:
        result = await session.exec(
            cashout_query(user_id).order_by(PaymentTransaction.created_at.desc())  # type: ignore
        )
        return [
            make_cashout_payment_transaction(payment_transaction, destination_payment_instrument, point_transaction)
            for payment_transaction, destination_payment_instrument, point_transaction in result.all()
        ]
