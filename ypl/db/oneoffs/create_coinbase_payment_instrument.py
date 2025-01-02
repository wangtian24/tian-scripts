import os
import uuid
from datetime import datetime

from sqlalchemy import Connection, select, update
from sqlmodel import Session

from ypl.db.payments import PaymentInstrument, PaymentInstrumentFacilitatorEnum, PaymentInstrumentIdentifierTypeEnum
from ypl.db.users import User


def add_coinbase_retail_wallet_payment_instrument(connection: Connection) -> None:
    """Add coinbase payment instrument for coinbase retail payouts."""
    # Use a default test address if environment variable is not set
    coinbase_address = os.getenv("COINBASE_RETAIL_WALLET_ADDRESS", "0x0000000000000000000000000000000000000000")

    with Session(connection) as session:
        system_user = session.exec(select(User.user_id).where(User.name == "SYSTEM")).first()

        if not system_user:
            raise ValueError("SYSTEM user not found")

        existing = session.exec(
            select(PaymentInstrument).where(
                PaymentInstrument.user_id == system_user[0],
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.COINBASE,
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
                PaymentInstrument.deleted_at.is_(None),
            )
        ).first()

        if existing:
            return

        # Create new payment instrument
        # It does not matter what address we use here as for prod, we will update the address in a separate script
        new_uuid = uuid.uuid4()
        payment_instrument = PaymentInstrument(
            payment_instrument_id=new_uuid,
            user_id=system_user[0],
            facilitator=PaymentInstrumentFacilitatorEnum.COINBASE,
            identifier_type=PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
            identifier=coinbase_address,
        )

        session.add(payment_instrument)
        session.commit()


def remove_coinbase_retail_wallet_payment_instrument(connection: Connection) -> None:
    """Remove the system coinbase payment instrument if it exists."""
    with Session(connection) as session:
        system_user = session.exec(select(User.user_id).where(User.name == "SYSTEM")).first()

        if not system_user:
            return

        existing = (
            session.exec(
                select(PaymentInstrument).where(
                    PaymentInstrument.user_id == system_user[0],
                    PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.COINBASE,
                    PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.CRYPTO_ADDRESS,
                )
            )
            .scalars()
            .first()
        )

        if existing:
            session.exec(
                update(PaymentInstrument)
                .where(PaymentInstrument.payment_instrument_id == existing.payment_instrument_id)
                .values(deleted_at=datetime.now())
            )
            session.commit()
