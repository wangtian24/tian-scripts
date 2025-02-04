import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException

from ypl.backend.db import get_async_session
from ypl.backend.llm.invite import (
    GetInviteCodesResponse,
    UpdateInviteCodeRequest,
    get_invite_codes_for_user,
    update_invite_code_for_user,
)
from ypl.backend.utils.json import json_dumps

router = APIRouter()


@router.get("/admin/users/{user_id}/invite-codes")
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


@router.patch("/admin/users/{user_id}/invite-codes/{invite_code_id}")
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
