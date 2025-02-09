import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, HTTPException, Path, Query
from sqlalchemy import String, cast
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, or_, select

from ypl.backend.db import get_async_session
from ypl.backend.payment.hyperwallet.hyperwallet_utils import UserTokensResponse, get_hyperwallet_user_tokens
from ypl.backend.user.user import (
    RegisterVendorRequest,
    UserSearchResponse,
    UserSearchResult,
    VendorProfileResponse,
    deactivate_user,
    get_all_users,
    register_user_with_vendor,
)
from ypl.backend.utils.json import json_dumps
from ypl.db.invite_codes import SpecialInviteCode, SpecialInviteCodeClaimLog
from ypl.db.payments import PaymentInstrument, PaymentTransaction
from ypl.db.users import User, VendorNameEnum, WaitlistedUser

router = APIRouter()
admin_router = APIRouter()


class RelationshipType(str, Enum):
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"


class RelationshipBasis(str, Enum):
    REFERRAL = "referral"


@dataclass
class RelatedUser:
    user_id: str
    name: str | None
    relationship_type: RelationshipType
    relationship_basis: RelationshipBasis
    points: int
    created_at: datetime | None


@dataclass
class RelatedUsersResponse:
    related_users: list[RelatedUser]
    num_children: int
    num_siblings: int


@router.post("/users/register_user_with_vendor", response_model=VendorProfileResponse)
async def register_user_with_vendor_route(request: RegisterVendorRequest) -> VendorProfileResponse:
    """Register a user with a vendor.

    Args:
        request: The request containing user_id, vendor_id and optional additional details

    Returns:
        The created user vendor profile
    """
    try:
        return await register_user_with_vendor(request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        log_dict = {
            "message": "Error registering the user with the vendor",
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/users/get-user-tokens")
async def get_user_tokens(user_id: str, vendor_name: str) -> UserTokensResponse:
    """Get a vendor token for a user.

    Args:
        user_id: The ID of the user to get the vendor token for
        vendor_name: The name of the vendor to get the token for
    Returns:
        The vendor token for the user
    """
    if vendor_name == VendorNameEnum.HYPERWALLET.value:
        return await get_hyperwallet_user_tokens(user_id)
    else:
        raise HTTPException(status_code=400, detail=f"Vendor {vendor_name} not supported")


@admin_router.get("/admin/users")
async def get_all_users_route(limit: int = 100, offset: int = 0) -> UserSearchResponse:
    """Get all users."""
    log_dict = {
        "message": "Getting all users",
        "limit": limit,
        "offset": offset,
    }
    logging.info(json_dumps(log_dict))
    return await get_all_users(limit, offset)


@admin_router.get("/admin/users/search")
async def get_users(query: str) -> UserSearchResponse:
    """Search for users where name or user_id partially matches the query string.

    Args:
        query: Search string to match against user name or ID

    Returns:
        List of matching users with their ID, name, email, created_at, deleted_at, points and status
    """
    log_dict = {
        "message": "Searching for users",
        "query": query,
    }
    logging.info(json_dumps(log_dict))
    try:
        async with get_async_session() as session:
            search = f"%{query}%"
            stmt = select(User).where(
                or_(
                    col(User.name).ilike(search),
                    col(User.user_id).ilike(search),
                    col(User.email).ilike(search),
                )
            )

            result = await session.execute(stmt)
            users = result.scalars().all()

            log_dict = {
                "message": "Users found for search query in users table",
                "query": query,
                "users_count": str(len(users)),
            }
            logging.info(json_dumps(log_dict))

            if len(users) > 0:
                return UserSearchResponse(
                    users=[
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
                        for user in users
                    ],
                    has_more_rows=False,
                )

            # if this is not a user related data point, look for payment instrument
            stmt = select(User).join(PaymentInstrument).where(col(PaymentInstrument.identifier).ilike(search))

            result = await session.execute(stmt)
            users = result.scalars().all()

            log_dict = {
                "message": "Users found for search query in payment instruments",
                "query": query,
                "users_count": str(len(users)),
            }
            logging.info(json_dumps(log_dict))

            if len(users) > 0:
                return UserSearchResponse(
                    users=[
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
                        for user in users
                    ],
                    has_more_rows=False,
                )

            # if this is not a payment instrument then search for payment transaction id
            stmt = (
                select(User)
                .join(PaymentInstrument, User.user_id == PaymentInstrument.user_id)  # type: ignore
                .join(
                    PaymentTransaction,
                    PaymentInstrument.payment_instrument_id == PaymentTransaction.destination_instrument_id,  # type: ignore
                )
                .where(cast(PaymentTransaction.payment_transaction_id, String).ilike(search))
            )
            result = await session.execute(stmt)
            users = result.scalars().all()

            log_dict = {
                "message": "Users found for search query in payment transactions",
                "query": query,
                "users_count": str(len(users)),
            }
            logging.info(json_dumps(log_dict))

            if len(users) > 0:
                return UserSearchResponse(
                    users=[
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
                        for user in users
                    ],
                    has_more_rows=False,
                )

            return UserSearchResponse(users=[], has_more_rows=False)
    except Exception as e:
        log_dict = {
            "message": "Error searching for users",
            "query": query,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@admin_router.get("/admin/users/{user_id}/related")
async def get_related_users(user_id: str = Path(..., description="User ID")) -> RelatedUsersResponse:
    """Get users related to the given user.

    Returns a list of related users with:
    - user_id: User ID
    - name: User's name
    - relationship_type: Type of relationship (parent/child)
    - relationship_basis: Basis of relationship
    """
    log_dict = {
        "message": "Getting related users",
        "user_id": user_id,
    }
    logging.info(json_dumps(log_dict))
    try:
        async with get_async_session() as session:
            parent = await _get_parent_user(session, user_id)
            siblings = await _get_sibling_users(session, user_id, parent.user_id if parent else None)
            children = await _get_children_users(session, user_id)

            related_users = []
            if parent:
                related_users.append(parent)
                related_users.extend(siblings)
            related_users.extend(children)
            log_dict = {
                "message": "Related users found",
                "user_id": user_id,
                "related_users_count_including_siblings": str(len(related_users)),
                "parent_count": str(
                    len([user for user in related_users if user.relationship_type == RelationshipType.PARENT])
                ),
                "child_count": str(
                    len([user for user in related_users if user.relationship_type == RelationshipType.CHILD])
                ),
                "sibling_count": str(
                    len([user for user in related_users if user.relationship_type == RelationshipType.SIBLING])
                ),
            }
            logging.info(json_dumps(log_dict))

            return RelatedUsersResponse(
                related_users=related_users,
                num_children=len(children),
                num_siblings=len(siblings),
            )

    except Exception as e:
        log_dict = {
            "message": "Error getting related users",
            "user_id": user_id,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _get_parent_user(session: AsyncSession, user_id: str) -> RelatedUser | None:
    """Get the user who referred this user (parent) or created this user."""
    try:
        # First try to find referral parent
        log_dict = {
            "message": "Getting referral parent from special invite code claim log",
            "user_id": user_id,
        }
        logging.info(json_dumps(log_dict))
        query = (
            select(User)
            .select_from(SpecialInviteCodeClaimLog)
            .join(
                SpecialInviteCode,
                SpecialInviteCodeClaimLog.special_invite_code_id == SpecialInviteCode.special_invite_code_id,  # type: ignore
            )
            .join(User, User.user_id == SpecialInviteCode.creator_user_id)  # type: ignore
            .where(SpecialInviteCodeClaimLog.user_id == user_id)
        )

        result = await session.execute(query)
        parent = result.scalar_one_or_none()

        # Check presence in waitlisted_users table for referrer_id
        if not parent:
            log_dict = {
                "message": "Getting referral parent from waitlisted_users table",
                "user_id": user_id,
            }
            logging.info(json_dumps(log_dict))
            user_query = select(User).where(User.user_id == user_id)
            result = await session.execute(user_query)
            wu_user = result.scalar_one_or_none()

            if wu_user:
                waitlisted_query = (
                    select(User)
                    .select_from(WaitlistedUser)
                    .join(User, User.user_id == WaitlistedUser.referrer_id)  # type: ignore
                    .where(WaitlistedUser.email == wu_user.email)
                )
                result = await session.execute(waitlisted_query)
                parent = result.scalar_one_or_none()

        if not parent:
            # If no referral parent found, check for creator_user_id
            # First get the user to find their creator_user_id
            log_dict = {
                "message": "Getting creator user from users table",
                "user_id": user_id,
            }
            logging.info(json_dumps(log_dict))
            user_query = select(User).where(User.user_id == user_id)
            result = await session.execute(user_query)
            user = result.scalar_one_or_none()

            if user and user.creator_user_id:
                # Then get the creator user's details
                creator_query = select(User).where(User.user_id == user.creator_user_id)
                result = await session.execute(creator_query)
                parent = result.scalar_one_or_none()

        if not parent:
            return None

        return RelatedUser(
            user_id=parent.user_id,
            name=parent.name,
            relationship_type=RelationshipType.PARENT,
            relationship_basis=RelationshipBasis.REFERRAL,
            points=parent.points,
            created_at=parent.created_at,
        )
    except Exception as e:
        log_dict = {
            "message": "Error getting referral parent",
            "user_id": user_id,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        return None


async def _get_children_users(session: AsyncSession, user_id: str) -> list[RelatedUser]:
    """Get users that this user has referred (children)."""
    try:
        query = (
            select(User)
            .select_from(SpecialInviteCodeClaimLog)
            .join(
                SpecialInviteCode,
                SpecialInviteCode.special_invite_code_id == SpecialInviteCodeClaimLog.special_invite_code_id,  # type: ignore
            )
            .join(
                User,
                User.user_id == SpecialInviteCodeClaimLog.user_id,  # type: ignore
            )
            .where(SpecialInviteCode.creator_user_id == user_id)
        )

        result = await session.execute(query)
        children = result.scalars().all()

        # Also get users directly created by this user
        created_query = select(User).where(User.creator_user_id == user_id)
        created_result = await session.execute(created_query)
        created_children = created_result.scalars().all()

        log_dict = {
            "message": "Referred children found",
            "user_id": user_id,
            "referred_children_count": len(children),
        }
        logging.info(json_dumps(log_dict))

        log_dict = {
            "message": "Created children found",
            "user_id": user_id,
            "created_children_count": len(created_children),
        }
        logging.info(json_dumps(log_dict))

        # Combine both lists and remove duplicates
        all_children = {
            child.user_id: RelatedUser(
                user_id=child.user_id,
                name=child.name,
                relationship_type=RelationshipType.CHILD,
                relationship_basis=RelationshipBasis.REFERRAL,
                points=child.points,
                created_at=child.created_at,
            )
            for child in list(children) + list(created_children)
        }

        return list(all_children.values())
    except Exception as e:
        log_dict = {
            "message": "Error getting children",
            "user_id": user_id,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        return []


async def _get_sibling_users(session: AsyncSession, user_id: str, parent_user_id: str | None) -> list[RelatedUser]:
    """Get users who share the same parent (referrer or creator) as this user."""
    if not parent_user_id:
        return []

    try:
        # Get all children of the parent through referrals (these are siblings)
        query = (
            select(User)
            .select_from(SpecialInviteCodeClaimLog)
            .join(
                SpecialInviteCode,
                SpecialInviteCode.special_invite_code_id == SpecialInviteCodeClaimLog.special_invite_code_id,  # type: ignore
            )
            .join(
                User,
                User.user_id == SpecialInviteCodeClaimLog.user_id,  # type: ignore
            )
            .where(SpecialInviteCode.creator_user_id == parent_user_id)
        )

        result = await session.execute(query)
        referred_users = result.scalars().all()

        referred_siblings = [
            RelatedUser(
                user_id=user.user_id,
                name=user.name,
                relationship_type=RelationshipType.SIBLING,
                relationship_basis=RelationshipBasis.REFERRAL,
                points=user.points,
                created_at=user.created_at,
            )
            for user in referred_users
        ]

        # Get all users created by the parent
        query = select(User).where(User.creator_user_id == parent_user_id)
        result = await session.execute(query)
        created_users = result.scalars().all()

        # Convert created users to RelatedUser format
        created_siblings = [
            RelatedUser(
                user_id=user.user_id,
                name=user.name,
                relationship_type=RelationshipType.SIBLING,
                relationship_basis=RelationshipBasis.REFERRAL,
                points=user.points,
                created_at=user.created_at,
            )
            for user in created_users
        ]

        # Combine both lists and remove duplicates based on user_id
        all_siblings = {sibling.user_id: sibling for sibling in referred_siblings + created_siblings}

        # Filter out the original user
        if user_id in all_siblings:
            del all_siblings[user_id]

        return list(all_siblings.values())
    except Exception as e:
        log_dict = {
            "message": "Error getting siblings",
            "user_id": user_id,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        return []


@admin_router.post("/admin/users/{user_id}/deactivate")
async def deactivate_user_route(
    user_id: str = Path(..., description="User ID"),
    creator_user_email: str = Query(..., description="Email of the user performing the deactivation"),
) -> None:
    """Deactivate a user by setting their status to DEACTIVATED.

    Args:
        user_id: The ID of the user to deactivate
        creator_user_email: Email of the user performing the deactivation
    """
    log_dict = {
        "message": "Deactivating user",
        "user_id": user_id,
        "creator_user_email": creator_user_email,
    }
    logging.info(json_dumps(log_dict))
    try:
        await deactivate_user(user_id, creator_user_email)
    except Exception as e:
        log_dict = {
            "message": "Error deactivating user",
            "user_id": user_id,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
