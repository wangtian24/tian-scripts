from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field

from ypl.backend.config import settings
from ypl.backend.user.user import CashoutOverrideConfig, CashoutOverrideRequest, create_cashout_override
from ypl.backend.utils.soul_utils import SoulPermission, validate_permissions
from ypl.db.users import (
    UserCapabilityStatus,
)

router = APIRouter()

SLACK_WEBHOOK_CASHOUT = settings.SLACK_WEBHOOK_CASHOUT


class CashoutOverrideConfigModel(BaseModel):
    first_time_limit: int | None = Field(None, description="Override for first time cashout limit")
    daily_count: int | None = Field(None, description="Override for daily cashout count limit")
    weekly_count: int | None = Field(None, description="Override for weekly cashout count limit")
    monthly_count: int | None = Field(None, description="Override for monthly cashout count limit")
    daily_credits: int | None = Field(None, description="Override for daily cashout credits limit")
    weekly_credits: int | None = Field(None, description="Override for weekly cashout credits limit")
    monthly_credits: int | None = Field(None, description="Override for monthly cashout credits limit")


class CashoutOverrideRequestModel(BaseModel):
    user_id: str
    creator_user_email: str
    status: UserCapabilityStatus
    reason: str
    override_config: CashoutOverrideConfigModel | None = None


async def validate_manage_cashout(
    x_creator_email: str | None = Header(None, alias="X-Creator-Email"),
) -> None:
    """Validate that the user has MANAGE_CASHOUT permission."""
    await validate_permissions([SoulPermission.MANAGE_CASHOUT], x_creator_email)


@router.post(
    "/admin/user-capability/cashout/override",
    dependencies=[Depends(validate_manage_cashout)],
)
async def create_cashout_override_route(request: CashoutOverrideRequestModel) -> str:
    """Create a cashout capability override for a user.

    Args:
        request: The cashout override request containing:
            - user_id: The ID of the user to create the override for
            - creator_user_id: The ID of the user creating the override
            - status: The status of the override (ENABLED/DISABLED)
            - reason: The reason for creating this override
            - override_config: Optional configuration parameters for cashout limits

    Returns:
        The ID of the created capability override
    """
    override_config = None
    if request.override_config:
        override_config = CashoutOverrideConfig(
            first_time_limit=request.override_config.first_time_limit,
            daily_count=request.override_config.daily_count,
            weekly_count=request.override_config.weekly_count,
            monthly_count=request.override_config.monthly_count,
            daily_credits=request.override_config.daily_credits,
            weekly_credits=request.override_config.weekly_credits,
            monthly_credits=request.override_config.monthly_credits,
        )

    service_request = CashoutOverrideRequest(
        user_id=request.user_id,
        creator_user_email=request.creator_user_email,
        status=request.status,
        reason=request.reason,
        override_config=override_config,
    )

    return await create_cashout_override(service_request)
