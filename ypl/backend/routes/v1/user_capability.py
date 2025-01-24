import logging
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import func, select

from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import CapabilityType
from ypl.db.users import (
    Capability,
    CapabilityStatus,
    UserCapabilityOverride,
    UserCapabilityStatus,
)

router = APIRouter()


class CashoutOverrideConfig(BaseModel):
    first_time_limit: int | None = Field(None, description="Override for first time cashout limit")
    daily_count: int | None = Field(None, description="Override for daily cashout count limit")
    weekly_count: int | None = Field(None, description="Override for weekly cashout count limit")
    monthly_count: int | None = Field(None, description="Override for monthly cashout count limit")
    daily_credits: int | None = Field(None, description="Override for daily cashout credits limit")
    weekly_credits: int | None = Field(None, description="Override for weekly cashout credits limit")
    monthly_credits: int | None = Field(None, description="Override for monthly cashout credits limit")


class CashoutOverrideRequest(BaseModel):
    user_id: str
    creator_user_id: str
    status: UserCapabilityStatus
    reason: str
    override_config: CashoutOverrideConfig | None = None


@router.post("/user-capability/cashout/override", tags=["admin"])
async def create_cashout_override(request: CashoutOverrideRequest) -> str:
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
    log_dict = {
        "message": "Creating cashout override",
        "user_id": request.user_id,
        "creator_user_id": request.creator_user_id,
        "status": str(request.status),
        "reason": request.reason,
        "override_config": str(request.override_config or ""),
    }
    logging.info(json_dumps(log_dict))
    if request.creator_user_id == "request.user_id":
        raise HTTPException(status_code=400, detail="Internal Error")
    try:
        async with get_async_session() as session:
            capability_stmt = select(Capability).where(
                func.lower(Capability.capability_name) == CapabilityType.CASHOUT.value.lower(),
                Capability.deleted_at.is_(None),  # type: ignore
                Capability.status == CapabilityStatus.ACTIVE,
            )
            capability = (await session.exec(capability_stmt)).first()

            if not capability:
                log_dict = {
                    "message": "Error: Cashout capability not found",
                    "user_id": request.user_id,
                }
                logging.warning(json_dumps(log_dict))
                raise HTTPException(status_code=404, detail="Cashout capability not found")

            existing_overrides_stmt = select(UserCapabilityOverride).where(
                UserCapabilityOverride.user_id == request.user_id,
                UserCapabilityOverride.capability_id == capability.capability_id,
                UserCapabilityOverride.deleted_at.is_(None),  # type: ignore
            )
            existing_overrides = (await session.exec(existing_overrides_stmt)).all()
            for existing_override in existing_overrides:
                existing_override.deleted_at = datetime.now(UTC)
                existing_override.creator_user_id = request.creator_user_id

            override_config = None
            if request.override_config:
                override_config = {k: v for k, v in request.override_config.dict().items() if v is not None}
                if not override_config:
                    override_config = None

            override = UserCapabilityOverride(
                user_id=request.user_id,
                capability_id=capability.capability_id,
                creator_user_id=request.creator_user_id,
                status=request.status,
                reason=request.reason,
                effective_start_date=datetime.now(UTC),
                override_config=override_config,
            )

            session.add(override)
            await session.commit()
            await session.refresh(override)

            log_dict = {
                "message": "Successfully created cashout override",
                "user_id": request.user_id,
                "creator_user_id": request.creator_user_id,
                "status": str(request.status),
                "reason": request.reason,
                "override_config": str(override_config or ""),
            }
            logging.info(json_dumps(log_dict))

            return str(override.user_capability_override_id)

    except HTTPException:
        raise
    except Exception as e:
        log_dict = {
            "message": "Error creating cashout override",
            "user_id": request.user_id,
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
