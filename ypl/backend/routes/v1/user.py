import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, HTTPException, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, or_, select

from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.invite_codes import SpecialInviteCode, SpecialInviteCodeClaimLog
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


@dataclass
class RelatedUsersResponse:
    related_users: list[RelatedUser]


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
    try:
        async with get_async_session() as session:
            search = f"%{query}%"
            stmt = select(User).where(
                or_(
                    col(User.name).ilike(search),
                    col(User.user_id).ilike(search),
                )
            )

            result = await session.execute(stmt)
            users = result.scalars().all()

            log_dict = {
                "message": "Users found for search query",
                "query": query,
                "users_count": len(users),
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
                "related_users_count_including_siblings": len(related_users),
                "parent_count": len(
                    [user for user in related_users if user.relationship_type == RelationshipType.PARENT]
                ),
                "child_count": len(
                    [user for user in related_users if user.relationship_type == RelationshipType.CHILD]
                ),
                "sibling_count": len(
                    [user for user in related_users if user.relationship_type == RelationshipType.SIBLING]
                ),
            }
            logging.info(json_dumps(log_dict))

            return RelatedUsersResponse(related_users=related_users)

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
    )


async def _get_children_users(session: AsyncSession, user_id: str) -> list[RelatedUser]:
    """Get users that this user has referred (children)."""
    query = (
        select(User)
        .select_from(SpecialInviteCode)
        .join(
            SpecialInviteCodeClaimLog,
            SpecialInviteCodeClaimLog.special_invite_code_id == SpecialInviteCode.special_invite_code_id,  # type: ignore
        )
        .join(User, User.user_id == SpecialInviteCodeClaimLog.user_id)  # type: ignore
        .where(SpecialInviteCode.creator_user_id == user_id)
    )

    result = await session.execute(query)
    children = result.scalars().all()

    return [
        RelatedUser(
            user_id=child.user_id,
            name=child.name,
            relationship_type=RelationshipType.CHILD,
            relationship_basis=RelationshipBasis.REFERRAL,
        )
        for child in children
    ]


async def _get_sibling_users(session: AsyncSession, user_id: str, parent_user_id: str | None) -> list[RelatedUser]:
    """Get users who share the same parent (referrer or creator) as this user."""
    if not parent_user_id:
        return []

    # Get all children of the parent (these are siblings)
    siblings = await _get_children_users(session, parent_user_id)

    # Filter out the original user and convert children to siblings
    return [
        RelatedUser(
            user_id=sibling.user_id,
            name=sibling.name,
            relationship_type=RelationshipType.SIBLING,
            relationship_basis=RelationshipBasis.REFERRAL,
        )
        for sibling in siblings
        if sibling.user_id != user_id
    ]
