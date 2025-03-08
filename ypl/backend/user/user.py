import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import cast as type_cast
from uuid import UUID

import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy import String, and_, func, or_, select, true
from sqlalchemy.exc import DatabaseError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import BinaryExpression
from sqlmodel import col
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack_with_user_name
from ypl.backend.user.vendor_details import AdditionalDetails
from ypl.backend.user.vendor_registration import VendorRegistrationError, get_vendor_registration
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import CapabilityType
from ypl.db.chats import Turn
from ypl.db.invite_codes import SpecialInviteCode, SpecialInviteCodeClaimLog, SpecialInviteCodeState
from ypl.db.payments import PaymentInstrument, PaymentTransaction
from ypl.db.users import (
    Capability,
    CapabilityStatus,
    User,
    UserCapabilityOverride,
    UserCapabilityStatus,
    UserIPDetails,
    UserStatus,
    UserVendorProfile,
    VendorNameEnum,
)


@dataclass
class UserSearchResult:
    user_id: str
    name: str | None
    email: str
    created_at: datetime | None
    deleted_at: datetime | None
    points: int
    status: UserStatus
    discord_id: str | None
    discord_username: str | None
    image_url: str | None
    cashout_allowed: bool
    turn_count: int | None


@dataclass
class UserSearchResponse:
    users: list[UserSearchResult]
    has_more_rows: bool


@dataclass
class RegisterVendorRequest:
    user_id: str
    vendor_name: str
    additional_details: dict[str, Any] | None = None


@dataclass
class VendorProfileResponse:
    user_vendor_profile_id: UUID
    user_id: str
    user_vendor_id: str
    vendor_name: str
    additional_details: dict[str, Any] | None
    created_at: datetime | None
    modified_at: datetime | None
    deleted_at: datetime | None
    vendor_url_link: str | None = None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
)
async def register_user_with_vendor(request: RegisterVendorRequest) -> VendorProfileResponse:
    """Register a user with a vendor by creating a user vendor profile.

    Args:
        request: The request containing user_id, vendor_name and optional additional details

    Returns:
        The created user vendor profile

    Raises:
        ValueError: If the user is not found or vendor is not supported
        VendorRegistrationError: If vendor registration fails
    """
    log_dict: dict[str, Any] = {
        "message": "Registering user with vendor",
        "user_id": request.user_id,
        "vendor_name": request.vendor_name,
        "additional_details": request.additional_details,
    }
    logging.info(json_dumps(log_dict))

    try:
        async with get_async_session() as session:
            user_stmt = select(User).where(
                User.user_id == request.user_id,  # type: ignore
                User.deleted_at.is_(None),  # type: ignore
            )
            user = (await session.execute(user_stmt)).scalar_one_or_none()

            if not user:
                log_dict = {
                    "message": "Error: User not found",
                    "user_id": request.user_id,
                }
                logging.warning(json_dumps(log_dict))
                raise ValueError("Internal error: Invalid user")

            existing_profile_stmt = select(UserVendorProfile).where(
                UserVendorProfile.user_id == request.user_id,  # type: ignore
                UserVendorProfile.vendor_name == request.vendor_name,  # type: ignore
                UserVendorProfile.deleted_at.is_(None),  # type: ignore
            )
            existing_profile = (await session.execute(existing_profile_stmt)).scalar_one_or_none()

            if existing_profile:
                log_dict = {
                    "message": "User is already registered with the vendor",
                    "user_id": request.user_id,
                    "vendor_name": request.vendor_name,
                    "profile_id": str(existing_profile.user_vendor_profile_id),
                }
                logging.info(json_dumps(log_dict))
                return VendorProfileResponse(
                    user_vendor_profile_id=existing_profile.user_vendor_profile_id,
                    user_id=existing_profile.user_id,
                    user_vendor_id=existing_profile.user_vendor_id,
                    vendor_name=existing_profile.vendor_name.value,
                    additional_details=existing_profile.additional_details,
                    created_at=existing_profile.created_at,
                    modified_at=existing_profile.modified_at,
                    deleted_at=existing_profile.deleted_at,
                )

            additional_details = AdditionalDetails(**request.additional_details) if request.additional_details else None
            if not additional_details:
                log_dict = {
                    "message": "Error: Additional details are required",
                    "user_id": request.user_id,
                    "vendor_name": request.vendor_name,
                }
                logging.warning(json_dumps(log_dict))
                raise ValueError("Additional details are required")

            vendor_registration = get_vendor_registration(request.vendor_name)
            vendor_response = await vendor_registration.register_user(
                user_id=request.user_id,
                additional_details=additional_details,
            )

            profile = UserVendorProfile(
                user_vendor_profile_id=uuid.uuid4(),
                user_id=request.user_id,
                vendor_name=VendorNameEnum(request.vendor_name.lower()),
                user_vendor_id=vendor_response.vendor_id,
                additional_details=vendor_response.additional_details,
            )

            session.add(profile)
            await session.commit()

            log_dict = {
                "message": "Successfully registered user with vendor",
                "user_id": request.user_id,
                "vendor_name": request.vendor_name,
                "profile_id": str(profile.user_vendor_profile_id),
                "vendor_id": vendor_response.vendor_id,
            }
            logging.info(json_dumps(log_dict))

            return VendorProfileResponse(
                user_vendor_profile_id=profile.user_vendor_profile_id,
                user_id=profile.user_id,
                user_vendor_id=profile.user_vendor_id,
                vendor_name=profile.vendor_name.value,
                additional_details=profile.additional_details,
                created_at=profile.created_at,
                modified_at=profile.modified_at,
                deleted_at=profile.deleted_at,
                vendor_url_link=vendor_response.vendor_url_link,
            )

    except VendorRegistrationError as e:
        log_dict = {
            "message": "Vendor registration error: Error registering user with vendor",
            "error": str(e),
            "user_id": request.user_id,
            "vendor_name": request.vendor_name,
        }
        logging.error(json_dumps(log_dict))
        raise VendorRegistrationError(str(e)) from e

    except Exception as e:
        log_dict = {
            "message": "General error registering user with vendor",
            "error": str(e),
            "user_id": request.user_id,
            "vendor_name": request.vendor_name,
        }
        logging.error(json_dumps(log_dict))
        raise VendorRegistrationError(str(e)) from e


async def get_user(user_id: str) -> User:
    async with get_async_session() as session:
        stmt = select(User).where(
            User.user_id == user_id,  # type: ignore
        )
        user = (await session.execute(stmt)).scalar_one_or_none()
        if not user:
            raise ValueError("User not found")

        return type_cast(User, user)


def _build_search_pattern(query: str) -> str:
    """Build the SQL LIKE pattern for searching."""
    return f"%{query}%"


def _build_base_user_query() -> Select:
    """Build the base query for user selection.

    Returns:
        SQLAlchemy select statement with basic user fields
    """
    return select(User)


def _build_turn_count_subquery() -> Any:
    """Build a subquery to count the number of turns for each user.

    Returns:
        SQLAlchemy select statement that counts turns per user
    """
    return (
        select(User.user_id, func.count(Turn.turn_id).label("turn_count"))  # type: ignore
        .outerjoin(Turn, and_(User.user_id == Turn.creator_user_id, Turn.deleted_at.is_(None)))  # type: ignore
        .group_by(User.user_id)
    ).subquery()


def _add_cashout_check(query: Select, capability: Capability | None) -> Select:
    """Add cashout capability check to the query.

    Args:
        query: The base query to extend
        capability: The cashout capability if it exists

    Returns:
        SQLAlchemy select statement with cashout check added
    """
    if not capability:
        return query.add_columns(true().label("cashout_allowed"))

    cashout_check = (
        select(1)
        .where(
            and_(
                UserCapabilityOverride.user_id == User.user_id,  # type: ignore
                UserCapabilityOverride.capability_id == capability.capability_id,  # type: ignore
                UserCapabilityOverride.deleted_at.is_(None),  # type: ignore
                UserCapabilityOverride.status == UserCapabilityStatus.DISABLED,  # type: ignore
            )
        )
        .exists()
    )
    return query.add_columns((~cashout_check).label("cashout_allowed"))


def _add_turn_count(query: Select) -> Select:
    """Add turn count to the query.

    Args:
        query: The base query to extend

    Returns:
        SQLAlchemy select statement with turn count added
    """
    turn_counts = _build_turn_count_subquery()
    return query.add_columns(turn_counts.c.turn_count).join(
        turn_counts, User.user_id == turn_counts.c.user_id, isouter=True
    )


def _build_base_query_with_additional_fields(capability: Capability | None) -> Select:
    """Build the complete base query with all necessary fields.

    Args:
        capability: The cashout capability if it exists

    Returns:
        SQLAlchemy select statement with user, cashout check and turn count
    """
    query = _build_base_user_query()
    query = _add_cashout_check(query, capability)
    query = _add_turn_count(query)
    return query


def _build_user_search_conditions(search_pattern: str) -> BinaryExpression:
    """Build the search conditions for the user table.

    Args:
        search_pattern: The pattern to search for

    Returns:
        SQLAlchemy OR clause with all search conditions
    """
    return or_(
        col(User.name).ilike(search_pattern),  # type: ignore
        col(User.user_id).ilike(search_pattern),
        col(User.email).ilike(search_pattern),
        col(User.discord_id).ilike(search_pattern),
        col(User.discord_username).ilike(search_pattern),
        col(User.city).ilike(search_pattern),
        col(User.country_code).ilike(search_pattern),
        col(User.educational_institution).ilike(search_pattern),
        *(
            [func.lower(sa.cast(col(User.status), sa.String)) == func.lower(search_pattern.strip("%"))]
            if search_pattern.strip("%").upper() in [e.name for e in UserStatus]
            else []
        ),
    )


async def _execute_search_query(
    session: AsyncSession, base_query: Any, additional_conditions: Any, query: str, search_type: str
) -> UserSearchResponse | None:
    """Execute a search query and return results if found.

    Args:
        session: The database session
        base_query: The base query to execute
        additional_conditions: Additional WHERE conditions
        query: The original search query from the soul user
        search_type: The type of search being performed for logging

    Returns:
        UserSearchResponse if results found, None otherwise
    """
    stmt = base_query.where(additional_conditions)
    result = await session.execute(stmt)
    users_with_cashout = result.all()

    log_dict = {
        "message": f"Users found for search query in {search_type}",
        "query": query,
        "users_count": str(len(users_with_cashout)),
    }
    logging.info(json_dumps(log_dict))

    if len(users_with_cashout) > 0:
        return _create_user_search_response_with_additional_fields(users_with_cashout)
    return None


async def get_users(query: str) -> UserSearchResponse:
    """Search for users to support universal search.

    Args:
        query: Search string to match against various attributes of a user

    Returns:
        List of matching users with their ID, name, email, created_at, deleted_at, points, status,
        discord_id, discord_username, image_url, cashout_allowed, and turn_count

    Raises:
        HTTPException: If there's an error executing the search
    """
    log_dict = {
        "message": "Searching for users",
        "query": query,
    }
    logging.info(json_dumps(log_dict))

    try:
        async with get_async_session() as session:
            # Get cashout capability
            capability = await _get_cashout_capability(session)

            # Build base query and search pattern
            base_query = _build_base_query_with_additional_fields(capability)
            search_pattern = _build_search_pattern(query)

            # Try each search type in sequence
            # 1. Search in users table
            result = await _execute_search_query(
                session, base_query, _build_user_search_conditions(search_pattern), query, "users table"
            )
            if result:
                return result

            # 2. Search in payment instruments
            result = await _execute_search_query(
                session,
                base_query.join(PaymentInstrument),
                col(PaymentInstrument.identifier).ilike(search_pattern),
                query,
                "payment instruments",
            )
            if result:
                return result

            # 3. Search in payment transactions
            result = await _execute_search_query(
                session,
                base_query.join(PaymentInstrument, User.user_id == PaymentInstrument.user_id).join(  # type: ignore
                    PaymentTransaction,
                    PaymentInstrument.payment_instrument_id == PaymentTransaction.destination_instrument_id,  # type: ignore
                ),
                func.cast(col(PaymentTransaction.payment_transaction_id), String).ilike(search_pattern),
                query,
                "payment transactions",
            )
            if result:
                return result

            # 4. Search in IP addresses
            result = await _execute_search_query(
                session,
                base_query.join(UserIPDetails, User.user_id == UserIPDetails.user_id),  # type: ignore
                col(UserIPDetails.ip).ilike(search_pattern),
                query,
                "user IP details",
            )
            if result:
                return result

            # 5. Search in special invite codes
            result = await _execute_search_query(
                session,
                base_query.join(SpecialInviteCodeClaimLog).join(SpecialInviteCode),
                col(SpecialInviteCode.code).ilike(search_pattern),
                query,
                "special invite codes",
            )
            if result:
                return result

            return UserSearchResponse(users=[], has_more_rows=False)

    except SQLAlchemyError as e:
        log_dict = {
            "message": "Database error while searching for users",
            "query": query,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail="Database error occurred") from e
    except Exception as e:
        log_dict = {
            "message": "Error searching for users",
            "query": query,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _get_cashout_capability(session: AsyncSession) -> Capability | None:
    """Get the cashout capability if it exists.

    Args:
        session: The database session

    Returns:
        The cashout capability if it exists and is active
    """
    capability_stmt = select(Capability).where(
        Capability.capability_name == CapabilityType.CASHOUT.value,  # type: ignore
        Capability.deleted_at.is_(None),  # type: ignore
        Capability.status == CapabilityStatus.ACTIVE,  # type: ignore
    )
    return (await session.execute(capability_stmt)).scalar_one_or_none()


def _create_user_search_response_with_additional_fields(
    users_with_additional_fields: Sequence[Any],
) -> UserSearchResponse:
    """Create a UserSearchResponse from a list of users with their additional fields."""
    return UserSearchResponse(
        users=[
            UserSearchResult(
                user_id=row[0].user_id,
                name=row[0].name,
                email=row[0].email,
                created_at=row[0].created_at,
                deleted_at=row[0].deleted_at,
                points=row[0].points,
                status=row[0].status,
                discord_id=row[0].discord_id,
                discord_username=row[0].discord_username,
                image_url=row[0].image,
                cashout_allowed=True if row[1] else False,
                turn_count=row[2] if row[2] is not None else 0,
            )
            for row in users_with_additional_fields
        ],
        has_more_rows=False,
    )


async def get_all_users(limit: int = 100, offset: int = 0) -> UserSearchResponse:
    async with get_async_session() as session:
        # Get cashout capability
        capability = await _get_cashout_capability(session)

        # Build base query with all fields using our modular builders
        stmt = (
            _build_base_query_with_additional_fields(capability)
            .order_by(User.created_at.desc())  # type: ignore
            .limit(limit + 1)
            .offset(offset)
        )

        results = (await session.execute(stmt)).all()
        has_more_rows = len(results) > limit
        user_search_results = []
        for result in results[:limit]:
            user = result[0]  # User object
            cashout_allowed = result[1]  # cashout_allowed value
            turn_count = result[2]  # turn_count value
            user_search_results.append(
                UserSearchResult(
                    user_id=user.user_id,
                    name=user.name,
                    email=user.email,
                    created_at=user.created_at,
                    deleted_at=user.deleted_at,
                    points=user.points,
                    status=user.status,
                    discord_id=user.discord_id,
                    discord_username=user.discord_username,
                    image_url=user.image,
                    cashout_allowed=cashout_allowed,
                    turn_count=turn_count if turn_count is not None else 0,
                )
            )
        return UserSearchResponse(users=user_search_results, has_more_rows=has_more_rows)


async def _deactivate_user_invite_codes_status(session: AsyncSession, user_id: str) -> list[SpecialInviteCode]:
    """
    Returns:
        List of updated invite codes
    """
    query = select(SpecialInviteCode).where(SpecialInviteCode.creator_user_id == user_id)  # type: ignore

    result = await session.execute(query)
    invite_codes = list(result.scalars().all())
    for invite_code in invite_codes:
        invite_code.state = SpecialInviteCodeState.INACTIVE

    return invite_codes


async def reactivate_user(user_id: str, creator_user_email: str) -> None:
    try:
        await validate_not_self_action(user_id, creator_user_email)

        async with get_async_session() as session:
            stmt = select(User).where(User.user_id == user_id)  # type: ignore
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()

            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            user.status = UserStatus.ACTIVE
            user.deleted_at = None

            await session.commit()

            log_dict = {
                "message": "User reactivated successfully",
                "user_id": user_id,
                "creator_user_email": creator_user_email,
            }
            logging.info(json_dumps(log_dict))
    except Exception as e:
        log_dict = {
            "message": "Error reactivating user",
            "user_id": user_id,
            "creator_user_email": creator_user_email,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


async def deactivate_user(user_id: str, creator_user_email: str) -> None:
    try:
        await validate_not_self_action(user_id, creator_user_email)

        async with get_async_session() as session:
            stmt = select(User).where(User.user_id == user_id)  # type: ignore
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()

            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            # Update user status
            user.status = UserStatus.DEACTIVATED
            user.deleted_at = datetime.now()

            # Update all active invite codes to inactive
            invite_codes = await _deactivate_user_invite_codes_status(session, user_id)

            await session.commit()

            log_dict = {
                "message": "User deactivated successfully",
                "user_id": user_id,
                "creator_user_email": creator_user_email,
                "deactivated_invite_codes": "\n".join([invite_code.code for invite_code in invite_codes]),
            }
            logging.info(json_dumps(log_dict))

    except Exception as e:
        log_dict = {
            "message": "Error deactivating user",
            "user_id": user_id,
            "creator_user_email": creator_user_email,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


async def validate_not_self_action(user_id: str, creator_user_email: str) -> None:
    """Validate that a user is not trying to perform an action on themselves.

    Args:
        user_id: The ID of the user the action is being performed on
        creator_user_email: Email of the user performing the action

    Raises:
        HTTPException: If the creator is trying to perform an action on themselves
    """

    if user_id is None or creator_user_email is None:
        raise HTTPException(status_code=400, detail="User ID and creator user email are required")

    async with get_async_session() as session:
        creator_stmt = select(User).where(
            func.lower(User.email) == func.lower(creator_user_email),
            User.deleted_at.is_(None),  # type: ignore
        )
        creator = (await session.execute(creator_stmt)).scalar_one_or_none()

        if not creator:
            log_dict = {
                "message": "Error: Creator user not found",
                "creator_user_email": creator_user_email,
            }
            logging.warning(json_dumps(log_dict))
            raise HTTPException(status_code=404, detail="Creator user not found")

        if creator.user_id == user_id:
            log_dict = {
                "message": "Error: User cannot perform action on themselves",
                "user_id": user_id,
                "creator_user_email": creator_user_email,
            }
            logging.error(json_dumps(log_dict))
            await post_to_slack_with_user_name(user_id, json_dumps(log_dict))
            raise HTTPException(status_code=400, detail="Users cannot perform this action on themselves")
