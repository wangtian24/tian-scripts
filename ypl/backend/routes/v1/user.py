import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, HTTPException, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, func, or_, select

from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.invite_codes import SpecialInviteCode, SpecialInviteCodeClaimLog
from ypl.db.payments import PaymentInstrument
from ypl.db.users import User, UserStatus

router = APIRouter()


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


@dataclass
class UserSearchResult:
    user_id: str
    name: str | None
    email: str
    created_at: datetime | None
    deleted_at: datetime | None
    points: int
    status: UserStatus


@dataclass
class UserSearchResponse:
    users: list[UserSearchResult]


@router.get("/users/search")
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
                        )
                        for user in users
                    ]
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
                    )
                    for user in users
                ]
            )

    except Exception as e:
        log_dict = {
            "message": "Error searching for users",
            "query": query,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/users/{user_id}/related")
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
    # First try to find referral parent
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

    if not parent:
        # If no referral parent found, check for creator_user_id
        # First get the user to find their creator_user_id
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


async def _get_children_users(session: AsyncSession, user_id: str) -> list[RelatedUser]:
    """Get users that this user has referred (children)."""
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


async def _get_sibling_users(session: AsyncSession, user_id: str, parent_user_id: str | None) -> list[RelatedUser]:
    """Get users who share the same parent (referrer or creator) as this user."""
    if not parent_user_id:
        return []

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


@router.post("/users/{user_id}/deactivate")
async def deactivate_user(
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
                    "message": "Error: User cannot deactivate themselves",
                    "user_id": user_id,
                    "creator_user_email": creator_user_email,
                }
                logging.error(json_dumps(log_dict))
                raise HTTPException(status_code=400, detail="Users cannot deactivate themselves")

            stmt = select(User).where(User.user_id == user_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()

            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            user.status = UserStatus.DEACTIVATED
            user.deleted_at = datetime.now()
            await session.commit()

            log_dict = {
                "message": "User deactivated successfully",
                "user_id": user_id,
                "creator_user_email": creator_user_email,
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
