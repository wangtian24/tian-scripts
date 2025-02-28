import logging
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from functools import lru_cache

import email_normalize
from rapidfuzz.distance import DamerauLevenshtein
from sqlalchemy.orm import joinedload
from sqlmodel import select

from ypl.backend.abuse.utils import create_abuse_event, get_recent_users, get_referring_user, ip_details_str
from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.abuse import AbuseActionType, AbuseEventType
from ypl.db.users import User

EMAIL_NORMALIZER: email_normalize.Normalizer | None = None

MAX_USER_CREATION_TIME_FOR_EMAIL_OR_NAME_SIMILARITY_CHECK = timedelta(days=5)

EMAIL_SIMILARITY_THRESHOLD_RECENT_USER = 0.8
EMAIL_SIMILARITY_THRESHOLD_REFERRER = 0.7
NAME_SIMILARITY_THRESHOLD_RECENT_USER = 0.9
NAME_SIMILARITY_THRESHOLD_REFERRER = 0.6
EMAIL_SIMILARITY_MIN_LENGTH = 6
NAME_SIMILARITY_MIN_LENGTH = 6


@lru_cache(maxsize=1000)
async def normalize_email(email_address: str) -> str:
    global EMAIL_NORMALIZER
    if not EMAIL_NORMALIZER:
        EMAIL_NORMALIZER = email_normalize.Normalizer()
    result = await EMAIL_NORMALIZER.normalize(email_address)
    return result.normalized_address if result and result.normalized_address else email_address  # type: ignore


def simple_normalize_email(email: str) -> str:
    name, domain = email.split("@")
    # Remove the section between the "+" and the "@" symbol.
    name = re.sub(r"\+.*", "", name)
    # Lowercase and remove periods.
    name = name.lower().replace(".", "")
    return f"{name}@{domain}"


def normalize_email_for_similarity_check(email: str) -> str:
    email = simple_normalize_email(email)
    # Remove trailing digits.
    return re.sub(r"\d+$", "", email.split("@")[0])


def normalize_name_for_similarity_check(name: str) -> str:
    # Lowercase, remove consecutive and trailing spaces.
    return " ".join(name.lower().strip().split())


def _find_similar_strings(
    string: str,
    other_strings: list[str],
    user_ids: list[str],
    min_similarity: float,
    min_length: int,
    normalize_func: Callable[[str], str] | None = None,
) -> dict[str, tuple[str, float]]:
    """Find similar strings to the src string, by edit distance.

    Args:
        src: The string to find similar strings to.
        other_strings: A list of strings to compare against.
        user_ids: A list of user IDs that correspond to the strings.
        min_similarity: The minimum similarity score; between 0 and 1.
        min_length: The minimum length of a (normalized) string to consider; shorter strings are excluded.
        normalize_func: A function to normalize the strings before comparing them.

    Returns:
        A dictionary of similar strings, with the matching user_id as the key and a tuple of the normalized string and
        its similarity score as the value.
    """
    similar_strings: dict[str, tuple[str, float]] = {}
    nrm_string = normalize_func(string) if normalize_func else string
    if len(nrm_string) < min_length:
        return similar_strings

    for other_string, user_id in zip(other_strings, user_ids, strict=True):
        nrm_other_string = normalize_func(other_string) if normalize_func else other_string
        if len(nrm_other_string) < min_length:
            continue
        similarity = DamerauLevenshtein.normalized_similarity(nrm_string, nrm_other_string)
        if similarity < min_similarity:
            continue
        similar_strings[user_id] = (other_string, similarity)
    return similar_strings


def _find_similar_emails(
    email: str,
    emails: list[str],
    user_ids: list[str],
    threshold: float,
) -> dict[str, tuple[str, float]]:
    return _find_similar_strings(
        string=email,
        other_strings=emails,
        user_ids=user_ids,
        min_similarity=threshold,
        min_length=EMAIL_SIMILARITY_MIN_LENGTH,
        normalize_func=normalize_email_for_similarity_check,
    )


def _find_similar_names(
    name: str,
    names: list[str],
    user_ids: list[str],
    threshold: float,
) -> dict[str, tuple[str, float]]:
    return _find_similar_strings(
        string=name,
        other_strings=names,
        user_ids=user_ids,
        min_similarity=threshold,
        min_length=NAME_SIMILARITY_MIN_LENGTH,
        normalize_func=normalize_name_for_similarity_check,
    )


async def check_similar_recent_signups_abuse(user_id: str) -> None:
    """Check for recently signed-up users with similar credentials."""

    async with get_async_session() as session:
        new_user = (
            await session.exec(select(User).options(joinedload(User.ip_details)).where(User.user_id == user_id))  # type: ignore
        ).first()
        if not new_user:
            logging.warning(json_dumps({"user_id": user_id, "message": "User not found"}))
            return

        ip_details = new_user.ip_details
        name = new_user.name
        email = simple_normalize_email(new_user.email)

        event_details = {
            "user_id": new_user.user_id,
            "user_name": name,
            "user_email": f"{new_user.email} (normalized: {email})",
            "user_ip_details": ip_details_str(ip_details),
            "user_created_at": str(new_user.created_at),
        }
        alerted_user_ids: set[str] = set()

        # Check for exact email matches to any existing users.
        same_email_users = (await session.exec(select(User).where(User.email == email, User.user_id != user_id))).all()
        if same_email_users:
            await create_abuse_event(
                session,
                new_user,
                AbuseEventType.SIGNUP_SAME_EMAIL_AS_EXISTING_USER,
                event_details=event_details
                | {
                    "existing_users": [
                        {
                            "user_id": user.user_id,
                            "user_name": user.name,
                            "user_email": user.email,
                            "user_created_at": str(user.created_at),
                        }
                        for user in same_email_users
                    ],
                },
                actions={AbuseActionType.SLACK_REPORT},
            )
            alerted_user_ids.update([user.user_id for user in same_email_users])

        # Check for near-matches of email/name to the referring account.
        referring_user = await get_referring_user(session, user_id)
        if referring_user and referring_user.user_id not in alerted_user_ids:
            referrer_event_details = event_details | {
                "referrer_user_id": referring_user.user_id,
                "referrer_user_name": referring_user.name,
                "referrer_user_email": referring_user.email,
                "referrer_user_created_at": str(referring_user.created_at),
                "referrer_user_ip_details": ip_details_str(referring_user.ip_details),
            }
            similar_emails = _find_similar_emails(
                email, [referring_user.email], [referring_user.user_id], threshold=EMAIL_SIMILARITY_THRESHOLD_REFERRER
            )
            if similar_emails:
                await create_abuse_event(
                    session,
                    new_user,
                    AbuseEventType.SIGNUP_SIMILAR_EMAIL_AS_REFERRER,
                    event_details=referrer_event_details,
                    actions={AbuseActionType.SLACK_REPORT},
                )
                alerted_user_ids.add(referring_user.user_id)
            if name is not None and referring_user.name is not None and referring_user.user_id not in alerted_user_ids:
                similar_names = _find_similar_names(
                    name, [referring_user.name], [referring_user.user_id], threshold=NAME_SIMILARITY_THRESHOLD_REFERRER
                )
                if similar_names:
                    await create_abuse_event(
                        session,
                        new_user,
                        AbuseEventType.SIGNUP_SIMILAR_NAME_AS_REFERRER,
                        event_details=referrer_event_details,
                        actions={AbuseActionType.SLACK_REPORT},
                    )
                    alerted_user_ids.add(referring_user.user_id)
        # Check for near-matches of email/name to recent users.
        min_creation_time: datetime = new_user.created_at - MAX_USER_CREATION_TIME_FOR_EMAIL_OR_NAME_SIMILARITY_CHECK  # type: ignore
        recent_users = await get_recent_users(session, min_creation_time=min_creation_time)
        recent_users = [
            user for user in recent_users if user.user_id != user_id and user.user_id not in alerted_user_ids
        ]
        recent_users_by_id = {user.user_id: user for user in recent_users}
        recent_user_emails = [user.email for user in recent_users]
        recent_user_names = [user.name if user.name is not None else "" for user in recent_users]
        similar_emails = _find_similar_emails(
            email, recent_user_emails, list(recent_users_by_id.keys()), threshold=EMAIL_SIMILARITY_THRESHOLD_RECENT_USER
        )
        if similar_emails:
            similar_users = [
                {
                    "user_id": similar_user_id,
                    "user_name": recent_users_by_id[similar_user_id].name,
                    "user_email": f"{recent_users_by_id[similar_user_id].email} (normalized: {similar_user_email})",
                    "email_similarity_score": similarity,
                    "user_created_at": str(recent_users_by_id[similar_user_id].created_at),
                }
                for similar_user_id, (similar_user_email, similarity) in similar_emails.items()
            ]
            await create_abuse_event(
                session,
                new_user,
                AbuseEventType.SIGNUP_SIMILAR_EMAIL_AS_RECENT_USER,
                event_details=event_details | {"similar_users": similar_users},
                actions={AbuseActionType.SLACK_REPORT},
            )
            alerted_user_ids.update([user.user_id for user in recent_users])
        if name is not None:
            similar_names = _find_similar_names(
                name,
                recent_user_names,
                list(recent_users_by_id.keys()),
                threshold=NAME_SIMILARITY_THRESHOLD_RECENT_USER,
            )
            if similar_names:
                similar_users = [
                    {
                        "user_id": similar_user_id,
                        "user_name": recent_users_by_id[similar_user_id].name,
                        "user_email": recent_users_by_id[similar_user_id].email,
                        "name_similarity_score": similarity,
                        "user_created_at": str(recent_users_by_id[similar_user_id].created_at),
                    }
                    for similar_user_id, (_, similarity) in similar_names.items()
                ]
                await create_abuse_event(
                    session,
                    new_user,
                    AbuseEventType.SIGNUP_SIMILAR_NAME_AS_RECENT_USER,
                    event_details=event_details | {"similar_users": similar_users},
                    actions={AbuseActionType.SLACK_REPORT},
                )
