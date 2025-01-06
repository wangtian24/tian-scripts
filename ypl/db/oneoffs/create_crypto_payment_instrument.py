import os
import uuid
from datetime import datetime

from sqlalchemy import Connection, insert, select, update
from sqlmodel import Session

from ypl.db.payments import PaymentInstrument, PaymentInstrumentFacilitatorEnum, PaymentInstrumentIdentifierTypeEnum
from ypl.db.users import User


def add_crypto_payment_instrument(connection: Connection) -> None:
    """Add crypto payment instrument for crypto rewards."""
    # Use a default test address if environment variable is not set
    crypto_address = os.getenv("CRYPTO_PAYMENT_INSTRUMENT_ADDRESS", "0x0000000000000000000000000000000000000000")

    with Session(connection) as session:
        system_user = session.exec(select(User.user_id).where(User.name == "SYSTEM")).first()

        if not system_user:
            raise ValueError("SYSTEM user not found")

        existing = session.exec(
            select(PaymentInstrument.payment_instrument_id).where(
                PaymentInstrument.user_id == system_user[0],
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.ON_CHAIN,
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
                PaymentInstrument.deleted_at.is_(None),
            )
        ).first()

        if existing:
            return

        # Create new payment instrument using a raw insert to control exactly which columns are included
        new_uuid = uuid.uuid4()
        stmt = insert(PaymentInstrument).values(
            payment_instrument_id=new_uuid,
            user_id=system_user[0],
            facilitator=PaymentInstrumentFacilitatorEnum.ON_CHAIN,
            identifier_type=PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
            identifier=crypto_address,
        )
        session.execute(stmt)
        session.commit()


def remove_crypto_payment_instrument(connection: Connection) -> None:
    """Remove the system payment instrument if it exists."""
    with Session(connection) as session:
        system_user = session.exec(select(User.user_id).where(User.name == "SYSTEM")).first()

        if not system_user:
            return

        existing = (
            session.exec(
                select(PaymentInstrument.payment_instrument_id).where(
                    PaymentInstrument.user_id == system_user[0],
                    PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.ON_CHAIN,
                    PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
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
