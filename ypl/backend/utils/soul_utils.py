import logging
from typing import cast
from uuid import UUID

from fastapi import Header, HTTPException
from sqlalchemy import select
from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.db.soul_rbac import RolePermission, SoulPermission, UserRole
from ypl.db.users import User


async def has_permission(creator_user_email: str, permission: SoulPermission) -> bool:
    """
    Check if a user has a specific permission based on their email.

    Args:
        session: The database session
        creator_user_email: Email of the user to check permissions for
        permission: The permission to check

    Returns:
        bool: True if user has the permission, False otherwise
    """
    try:
        async with get_async_session() as session:
            user_stmt = select(User).where(
                User.email == creator_user_email,  # type: ignore
                User.deleted_at.is_(None),  # type: ignore
            )
            result = await session.execute(user_stmt)
            user = result.scalar_one_or_none()
            if not user:
                return False

            # Get all roles for the user
            roles_stmt = select(UserRole.role_id).where(UserRole.user_id == user.user_id)  # type: ignore
            role_results = await session.execute(roles_stmt)
            user_role_ids: list[UUID] = [cast(UUID, r[0]) for r in role_results]
            if not user_role_ids:
                return False

            # Check if any of the user's roles have the required permission
            permission_stmt = select(RolePermission).where(
                RolePermission.role_id.in_(user_role_ids),  # type: ignore[attr-defined]
                RolePermission.permission == permission,  # type: ignore
            )
            has_perm = (await session.execute(permission_stmt)).first() is not None
            return has_perm

    except Exception as e:
        log_dict = {
            "message": "Error checking user permission",
            "creator_user_email": creator_user_email,
            "permission": permission.value,
            "error": str(e),
        }
        logging.warning(json_dumps(log_dict))
        return False


async def validate_permissions(
    permissions: list[SoulPermission],
    x_creator_email: str | None = Header(None, alias="X-Creator-Email"),
) -> None:
    """
    FastAPI dependency to validate permissions.
    Gets creator email from X-Creator-Email header and required permissions from security scopes.

    Usage:
        # For user management endpoints
        @router.get(
            "/users",
            dependencies=[Depends(validate_permissions)],
            security=[SecurityScopes(["READ_USERS"])]
        )

        # For payment management endpoints
        @router.post(
            "/payment-instruments",
            dependencies=[Depends(validate_permissions)],
            security=[SecurityScopes(["MANAGE_PAYMENT_INSTRUMENTS"])]
        )

        # For cashout management endpoints
        @router.post(
            "/cashout/approve",
            dependencies=[Depends(validate_permissions)],
            security=[SecurityScopes(["MANAGE_CASHOUT"])]
        )

    Args:
        security_scopes: Security scopes containing required permissions
        x_creator_email: Email of the user to validate, passed via X-Creator-Email header

    Raises:
        HTTPException: If creator email is missing or user doesn't have required permissions
    """
    if not x_creator_email:
        raise HTTPException(status_code=401, detail="X-Creator-Email header is required")

    if not permissions:
        # No permissions required
        return

    # Convert scope strings to SoulPermission enums
    required_permissions = [SoulPermission(permission) for permission in permissions]

    for permission in required_permissions:
        if not await has_permission(x_creator_email, permission):
            log_dict = {
                "message": "Permission denied",
                "creator_user_email": x_creator_email,
                "required_permission": permission.value,
            }
            logging.warning(json_dumps(log_dict))
            raise HTTPException(status_code=403, detail=f"Permission denied. Required permission: {permission.value}")


def get_soul_url(user_id: str) -> str:
    """
    Generate the Soul URL for a given user ID.
    """
    return f"https://yupp-soul.vercel.app/users?query={user_id}"
