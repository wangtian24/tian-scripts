import logging

from fastapi import APIRouter, HTTPException, Query

from ypl.backend.llm.profile import (
    UserProfileResponse,
    get_user_profile,
)

router = APIRouter()


@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(user_id: str = Query(..., description="User ID")) -> UserProfileResponse:
    try:
        profile: UserProfileResponse | None = await get_user_profile(user_id)
    except Exception as e:
        logging.error(f"Error fetching profile for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e

    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile not found for user {user_id}")

    return profile
