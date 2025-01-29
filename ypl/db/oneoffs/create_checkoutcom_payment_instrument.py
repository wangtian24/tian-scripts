import os
import uuid
from datetime import datetime

from sqlalchemy import Connection, insert, select, update
from sqlmodel import Session

from ypl.db.payments import PaymentInstrument, PaymentInstrumentFacilitatorEnum, PaymentInstrumentIdentifierTypeEnum
from ypl.db.users import User


def add_checkoutcom_source_payment_instrument(connection: Connection) -> None:
    """Add Checkout.com payment instrument for Checkout.com payouts."""
    checkoutcom_account_id = os.getenv("CHECKOUTCOM_ACCOUNT_ID", "0000000000000000000000000000000000000000")

    with Session(connection) as session:
        system_user = session.exec(select(User.user_id).where(User.name == "SYSTEM")).first()

        if not system_user:
            raise ValueError("SYSTEM user not found")

        existing = session.exec(
            select(PaymentInstrument.payment_instrument_id).where(
                PaymentInstrument.user_id == system_user[0],
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.CHECKOUT_COM,
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.PARTNER_IDENTIFIER,
                PaymentInstrument.deleted_at.is_(None),
            )
        ).first()

        if existing:
            return

        # Create new payment instrument
        new_uuid = uuid.uuid4()
        stmt = insert(PaymentInstrument).values(
            payment_instrument_id=new_uuid,
            user_id=system_user[0],
            facilitator=PaymentInstrumentFacilitatorEnum.CHECKOUT_COM,
            identifier_type=PaymentInstrumentIdentifierTypeEnum.PARTNER_IDENTIFIER,
            identifier=checkoutcom_account_id,
        )

        session.execute(stmt)
        session.commit()


def remove_checkoutcom_source_payment_instrument(connection: Connection) -> None:
    """Remove the system Checkout.com payment instrument if it exists."""
    with Session(connection) as session:
        system_user = session.exec(select(User.user_id).where(User.name == "SYSTEM")).first()

        if not system_user:
            return

        existing = (
            session.exec(
                select(PaymentInstrument.payment_instrument_id).where(
                    PaymentInstrument.user_id == system_user[0],
                    PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.CHECKOUT_COM,
                    PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.PARTNER_IDENTIFIER,
                    PaymentInstrument.deleted_at.is_(None),
                )
            )
            .scalars()
            .first()
        )

        if existing:
            session.exec(
                update(PaymentInstrument)
                .where(PaymentInstrument.payment_instrument_id == existing)
                .values(deleted_at=datetime.now())
            )
            session.commit()
