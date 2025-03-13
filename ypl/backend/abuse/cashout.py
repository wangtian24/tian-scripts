import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import joinedload
from sqlmodel import select

from ypl.backend.abuse.utils import (
    create_abuse_event,
    get_referred_users,
    get_referring_user,
    ip_details_str,
    is_impossible_travel,
)
from ypl.backend.db import get_async_session
from ypl.backend.utils.async_utils import create_background_task
from ypl.backend.utils.json import json_dumps
from ypl.db.abuse import AbuseActionType, AbuseEventType
from ypl.db.payments import PaymentInstrument, PaymentInstrumentIdentifierTypeEnum
from ypl.db.users import User

# Users sharing a payment instrument and created within less than this time are suspicious.
MIN_CREATION_TIME_BETWEEN_USERS_WITH_SAME_INSTRUMENT = timedelta(days=2)

# Users who cash out within this time of referring others (who signed up) are suspicious.
MIN_CREATION_TIME_BETWEEN_CASHOUT_AND_REFERRAL_SIGNUP = timedelta(days=1)

# Users who cash out and have The maximum number of recent referral signups that are allowed.
MAX_NUMBER_OF_RECENT_REFERRAL_SIGNUPS = 1


async def check_cashout_instrument_abuse(
    user_id: str,
    identifier_type: PaymentInstrumentIdentifierTypeEnum,
    identifier: str,
) -> None:
    """Check for related users sharing the same payment instrument as the cashing-out user."""

    async with get_async_session() as session:
        users_with_same_instrument = (
            await session.exec(
                select(User).where(
                    User.payment_instruments.any(  # type: ignore
                        (PaymentInstrument.identifier == identifier)
                        & (PaymentInstrument.identifier_type == identifier_type)
                    ),
                    User.deleted_at.is_(None),  # type: ignore
                    User.user_id != user_id,
                )
            )
        ).all()

        if not users_with_same_instrument:
            return

        cashing_out_user = (
            await session.exec(select(User).options(joinedload(User.ip_details)).where(User.user_id == user_id))  # type: ignore
        ).first()

        if not cashing_out_user:
            logging.warning(json_dumps({"user_id": user_id, "message": "User not found"}))
            return

        users_with_same_instrument_ids = set([user.user_id for user in users_with_same_instrument])
        referring_user = await get_referring_user(session, user_id)

        event_details = {
            "cashing_out_user_id": cashing_out_user.user_id,
            "cashing_out_user_email": cashing_out_user.email,
            "cashing_out_user_created_at": str(cashing_out_user.created_at),
            "cashing_out_user_ip_details": ip_details_str(cashing_out_user.ip_details),
            "payment_instrument_identifier": identifier,
            "payment_instrument_identifier_type": identifier_type.value,
        }

        # The referring user is using the same instrument as the cashing-out user.
        if referring_user and referring_user.user_id in users_with_same_instrument_ids:
            await create_abuse_event(
                session,
                cashing_out_user,
                AbuseEventType.CASHOUT_SAME_INSTRUMENT_AS_REFERRER,
                event_details=event_details
                | {
                    "referring_user_id": referring_user.user_id,
                    "referring_user_email": referring_user.email,
                    "referring_user_created_at": str(referring_user.created_at),
                },
                actions={AbuseActionType.SLACK_REPORT},
            )

        # There are other recent users with the same instrument.
        for same_instrument_user in users_with_same_instrument:
            delta = abs(cashing_out_user.created_at - same_instrument_user.created_at)  # type: ignore
            if delta < MIN_CREATION_TIME_BETWEEN_USERS_WITH_SAME_INSTRUMENT:
                await create_abuse_event(
                    session,
                    cashing_out_user,
                    AbuseEventType.CASHOUT_SAME_INSTRUMENT_AS_RECENT_NEW_USER,
                    event_details=event_details
                    | {
                        "recent_new_user_id": same_instrument_user.user_id,
                        "recent_new_user_email": same_instrument_user.email,
                        "recent_new_user_created_at": str(same_instrument_user.created_at),
                    },
                )


async def check_cashout_referral_abuse(user_id: str, cashout_time: datetime | None = None) -> None:
    """Check for excessive recent successful referrals from a cashing-out user, relative to `cashout_time` (or now)."""

    async with get_async_session() as session:
        cashing_out_user = (await session.exec(select(User).where(User.user_id == user_id))).first()
        if not cashing_out_user:
            logging.warning(json_dumps({"user_id": user_id, "message": "User not found"}))
            return

        event_details = {
            "cashing_out_user_id": cashing_out_user.user_id,
            "cashing_out_user_email": cashing_out_user.email,
            "cashing_out_user_created_at": str(cashing_out_user.created_at),
        }

        # Check for the number of recently signed-up users who were referred by the cashing-out user.
        cashout_time = cashout_time or datetime.now(UTC)
        referred_users = await get_referred_users(session, user_id)
        recent_referred_users = [
            user
            for user in referred_users
            if user.created_at
            and cashout_time - user.created_at < MIN_CREATION_TIME_BETWEEN_CASHOUT_AND_REFERRAL_SIGNUP
        ]

        if len(recent_referred_users) > MAX_NUMBER_OF_RECENT_REFERRAL_SIGNUPS:
            await create_abuse_event(
                session,
                cashing_out_user,
                AbuseEventType.CASHOUT_MULTIPLE_RECENT_REFERRAL_SIGNUPS,
                event_details=event_details
                | {
                    "recent_referred_user": [
                        {
                            "user_id": u.user_id,
                            "email": u.email,
                            "created_at": str(u.created_at),
                        }
                        for u in recent_referred_users
                    ]
                },
            )


async def check_cashout_travel_abuse(user_id: str, ip_address: str | None = None, is_crypto: bool = False) -> None:
    """Check for travel abuse by the cashing-out user."""
    if not is_crypto:
        return

    if ip_address:
        await is_impossible_travel(user_id, ip_address)


async def check_cashout_abuse(
    user_id: str,
    identifier_type: PaymentInstrumentIdentifierTypeEnum,
    identifier: str,
    is_crypto: bool,
    cashout_time: datetime | None = None,
    ip_address: str | None = None,
) -> None:
    create_background_task(check_cashout_instrument_abuse(user_id, identifier_type, identifier))
    create_background_task(check_cashout_referral_abuse(user_id, cashout_time=cashout_time))
    create_background_task(check_cashout_travel_abuse(user_id, ip_address=ip_address, is_crypto=is_crypto))
