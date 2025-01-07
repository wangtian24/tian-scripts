import uuid
from datetime import datetime

from sqlalchemy import Connection
from sqlmodel import Session, select, update

from ypl.backend.payment.upi.axis.facilitator import AxisUpiFacilitator
from ypl.db.payments import PaymentInstrument, PaymentInstrumentFacilitatorEnum, PaymentInstrumentIdentifierTypeEnum
from ypl.db.users import User


def add_axis_upi_payment_instrument(connection: Connection) -> None:
    """Add a UPI payment instrument for UPI payouts via our Axis bank account."""

    with Session(connection) as session:
        system_user = session.exec(select(User.user_id).where(User.name == "SYSTEM")).first()

        if not system_user:
            raise ValueError("SYSTEM user not found")

        existing_id = session.exec(
            select(PaymentInstrument.payment_instrument_id).where(
                PaymentInstrument.user_id == system_user,
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.UPI,
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.UPI_ID,
                PaymentInstrument.identifier == AxisUpiFacilitator.SOURCE_INSTRUMENT_UPI_ID,
            )
        ).first()

        if existing_id:
            session.exec(
                update(PaymentInstrument)
                .where(PaymentInstrument.payment_instrument_id == existing_id)
                .values(deleted_at=None)
            )
            session.commit()
            return

        payment_instrument = PaymentInstrument(
            payment_instrument_id=uuid.uuid4(),
            user_id=system_user,
            facilitator=PaymentInstrumentFacilitatorEnum.UPI,
            identifier_type=PaymentInstrumentIdentifierTypeEnum.UPI_ID,
            identifier=AxisUpiFacilitator.SOURCE_INSTRUMENT_UPI_ID,
        )

        session.add(payment_instrument)
        session.commit()


def remove_axis_upi_payment_instrument(connection: Connection) -> None:
    """Soft delete the system Axis UPI payment instrument if it exists."""
    with Session(connection) as session:
        system_user = session.exec(select(User.user_id).where(User.name == "SYSTEM")).first()

        if not system_user:
            return

        existing_id = session.exec(
            select(PaymentInstrument.payment_instrument_id).where(
                PaymentInstrument.user_id == system_user,
                PaymentInstrument.facilitator == PaymentInstrumentFacilitatorEnum.UPI,
                PaymentInstrument.identifier_type == PaymentInstrumentIdentifierTypeEnum.UPI_ID,
                PaymentInstrument.identifier == AxisUpiFacilitator.SOURCE_INSTRUMENT_UPI_ID,
            )
        ).first()

        if existing_id:
            session.exec(
                update(PaymentInstrument)
                .where(PaymentInstrument.payment_instrument_id == existing_id)
                .values(deleted_at=datetime.now())
            )
            session.commit()
