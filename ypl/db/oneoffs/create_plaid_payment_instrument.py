import os
import uuid
from datetime import datetime

from sqlalchemy import Connection, insert, select, update
from sqlmodel import Session

from ypl.db.payments import PaymentInstrument, PaymentInstrumentFacilitatorEnum, PaymentInstrumentIdentifierTypeEnum
from ypl.db.users import User


def add_plaid_payment_instrument(connection: Connection) -> None:
    """Add plaid payment instrument for plaid payouts."""
    # Use a default test account number if environment variable is not set
    plaid_account_number = os.getenv("PLAID_PAYMENT_INSTRUMENT_ACCOUNT", "000000000000")

    with Session(connection) as session:
        system_user = session.exec(select(User.user_id).where(User.name == "SYSTEM")).first()

        if not system_user:
            raise ValueError("SYSTEM user not found")

        existing = session.exec(
            select(PaymentInstrument.payment_instrument_id).where(
                PaymentInstrument.user_id == system_user[0],
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.PLAID,
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.BANK_ACCOUNT_NUMBER,
                PaymentInstrument.deleted_at.is_(None),
            )
        ).first()

        if existing:
            return

        # Create new payment instrument
        # It does not matter what account number we use here as for prod,
        # we will update the account in a separate script
        new_uuid = uuid.uuid4()
        stmt = insert(PaymentInstrument).values(
            payment_instrument_id=new_uuid,
            user_id=system_user[0],
            facilitator=PaymentInstrumentFacilitatorEnum.PLAID,
            identifier_type=PaymentInstrumentIdentifierTypeEnum.BANK_ACCOUNT_NUMBER,
            identifier=plaid_account_number,
        )

        session.execute(stmt)
        session.commit()


def remove_plaid_payment_instrument(connection: Connection) -> None:
    """Remove the system plaid payment instrument if it exists."""
    with Session(connection) as session:
        system_user = session.exec(select(User.user_id).where(User.name == "SYSTEM")).first()

        if not system_user:
            return

        existing = (
            session.exec(
                select(PaymentInstrument.payment_instrument_id).where(
                    PaymentInstrument.user_id == system_user[0],
                    PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.PLAID,
                    PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.BANK_ACCOUNT_NUMBER,
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
