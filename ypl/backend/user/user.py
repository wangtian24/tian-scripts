import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from ypl.backend.db import get_async_session
from ypl.backend.llm.utils import post_to_slack_with_user_name
from ypl.backend.user.vendor_details import AdditionalDetails
from ypl.backend.user.vendor_registration import VendorRegistrationError, get_vendor_registration
from ypl.backend.utils.json import json_dumps
from ypl.db.invite_codes import SpecialInviteCode, SpecialInviteCodeState
from ypl.db.users import User, UserStatus, UserVendorProfile, VendorNameEnum


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

        return cast(User, user)


async def get_all_users(limit: int = 100, offset: int = 0) -> UserSearchResponse:
    async with get_async_session() as session:
        stmt = (
            select(User)
            .order_by(User.created_at.desc())  # type: ignore
            .limit(limit + 1)
            .offset(offset)
        )
        users = (await session.execute(stmt)).scalars().all()
        has_more_rows = len(users) > limit
        user_search_results = []
        for user in users:
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
