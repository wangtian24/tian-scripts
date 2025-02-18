import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from ypl.backend.db import get_async_session
from ypl.backend.llm.invite import (
    GetInviteCodesResponse,
    UpdateInviteCodeRequest,
    create_invite_code_for_user,
    get_invite_codes_for_user,
    get_new_invite_code,
    update_invite_code_for_user,
)
from ypl.backend.utils.json import json_dumps

router = APIRouter()
admin_router = APIRouter()


@admin_router.get("/admin/users/{user_id}/invite-codes")
async def get_invite_codes(
    user_id: str,
) -> GetInviteCodesResponse:
    """Get invite codes for a user.

    Args:
        user_id: The ID of the user to get transactions for

    Returns:
        List of invite codes for this user with their usage statistics
    """
    try:
        async with get_async_session() as session:
            invite_codes = await get_invite_codes_for_user(user_id, session)
            return GetInviteCodesResponse(invite_codes=invite_codes)
    except Exception as e:
        log_dict = {
            "message": "Error getting invite codes",
            "user_id": user_id,
            "error": str(e),
        }
        logging.warning(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@admin_router.patch("/admin/users/{user_id}/invite-codes/{invite_code_id}")
async def update_invite_code(
    user_id: str,
    invite_code_id: UUID,
    update_request: UpdateInviteCodeRequest,
) -> None:
    """Update invite code status for a user.

    Args:
        user_id: The ID of the user who owns the invite code
        invite_code_id: The ID of the invite code to update
        update_request: The update request containing new values
    """
    try:
        async with get_async_session() as session:
            await update_invite_code_for_user(user_id, invite_code_id, update_request, session)
    except Exception as e:
        log_dict = {
            "message": "Error updating invite code",
            "user_id": user_id,
            "invite_code_id": invite_code_id,
            "error": str(e),
        }
        logging.warning(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@admin_router.post("/admin/invite-codes/generate")
async def generate_new_invite_code() -> str:
    """Generate a random invite code using word slugs without recording it in the database.

    Returns:
        A random invite code in the format "adjective-adjective-noun"
    """
    async with get_async_session() as session:
        return await get_new_invite_code(session)


class CreateInviteCodeRequest(BaseModel):
    code: str
    user_id: str
    usage_limit: int | None = 3
    referral_bonus_eligible: bool = True


@admin_router.post("/admin/users/{user_id}/invite-codes")
async def create_invite_code(request: CreateInviteCodeRequest) -> UUID:
    """Create a new invite code with the specified code string and creator user ID.

    Args:
        request: The request containing the code string, user ID, and optional parameters

    Returns:
        The UUID of the created invite code

    Raises:
        HTTPException: 409 if invite code already exists, 500 for other errors
    """
    try:
        async with get_async_session() as session:
            return await create_invite_code_for_user(
                code=request.code,
                user_id=request.user_id,
                session=session,
                usage_limit=request.usage_limit,
                referral_bonus_eligible=request.referral_bonus_eligible,
            )
    except IntegrityError as e:
        log_dict = {
            "message": "Invite code already exists",
            "code": request.code,
            "user_id": request.user_id,
        }
        logging.warning(json_dumps(log_dict))
        raise HTTPException(
            status_code=409,
            detail=f"Invite code '{request.code}' already exists",
        ) from e
    except Exception as e:
        log_dict = {
            "message": "Error creating invite code",
            "code": request.code,
            "user_id": request.user_id,
            "error": str(e),
        }
        logging.warning(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
