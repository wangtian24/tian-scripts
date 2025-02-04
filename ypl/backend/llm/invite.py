from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import desc, func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.db.invite_codes import (
    SpecialInviteCode,
    SpecialInviteCodeClaimLog,
    SpecialInviteCodeState,
)


@dataclass
class InviteCodeInfo:
    """Maps to SpecialInviteCode model."""

    code: str
    special_invite_code_id: UUID
    state: SpecialInviteCodeState
    usage_limit: int | None
    usage_last_7_days: int
    usage_last_30_days: int

    @classmethod
    def from_db_model(cls, model: SpecialInviteCode, usage_7d: int, usage_30d: int) -> "InviteCodeInfo":
        """Create InviteCodeInfo from a database model."""
        return cls(
            code=model.code,
            special_invite_code_id=model.special_invite_code_id,
            state=model.state,
            usage_limit=model.usage_limit,
            usage_last_7_days=usage_7d,
            usage_last_30_days=usage_30d,
        )


@dataclass
class GetInviteCodesResponse:
    invite_codes: list[InviteCodeInfo]


@dataclass
class UpdateInviteCodeRequest:
    """Request to update an invite code."""

    state: SpecialInviteCodeState | None = None
    usage_limit: int | None = None


async def get_invite_codes_for_user(user_id: str, session: AsyncSession) -> list[InviteCodeInfo]:
    """Get invite codes with usage statistics for a user.

    Args:
        user_id: The ID of the user to get invite codes for
        session: Database session

    Returns:
        List of invite codes with their usage statistics
    """
    now = datetime.now(UTC)
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    # Main query to get invite codes
    query = (
        select(SpecialInviteCode)
        .where(SpecialInviteCode.creator_user_id == user_id)
        .order_by(desc(SpecialInviteCode.created_at))  # type: ignore[arg-type]
    )

    result = await session.execute(query)
    invite_codes = result.scalars().all()

    if not invite_codes:
        return []

    # Get usage statistics for all codes.
    usage_query = (
        select(
            SpecialInviteCodeClaimLog.special_invite_code_id,
            func.count().filter(SpecialInviteCodeClaimLog.created_at >= seven_days_ago).label("usage_7d"),  # type: ignore
            func.count().filter(SpecialInviteCodeClaimLog.created_at >= thirty_days_ago).label("usage_30d"),  # type: ignore
        )
        .where(
            SpecialInviteCodeClaimLog.special_invite_code_id.in_([code.special_invite_code_id for code in invite_codes])  # type: ignore
        )
        .group_by(SpecialInviteCodeClaimLog.special_invite_code_id)  # type: ignore
    )

    usage_result = await session.execute(usage_query)
    usage_stats = {code_id: (usage_7d or 0, usage_30d or 0) for code_id, usage_7d, usage_30d in usage_result.all()}

    # Combine invite codes with their usage statistics
    invite_code_infos = []
    for code in invite_codes:
        usage_7d, usage_30d = usage_stats.get(code.special_invite_code_id, (0, 0))
        invite_code_infos.append(InviteCodeInfo.from_db_model(code, usage_7d, usage_30d))

    return invite_code_infos


async def update_invite_code_for_user(
    user_id: str,
    invite_code_id: UUID,
    update_request: UpdateInviteCodeRequest,
    session: AsyncSession,
) -> None:
    """Update invite code for a user.

    Args:
        user_id: The ID of the user who owns the invite code
        invite_code_id: The ID of the invite code to update
        update_request: The update request containing new values
        session: Database session

    Raises:
        ValueError: If the invite code doesn't exist or doesn't belong to the user
    """
    # Get the invite code
    query = select(SpecialInviteCode).where(
        SpecialInviteCode.special_invite_code_id == invite_code_id,
        SpecialInviteCode.creator_user_id == user_id,
    )
    result = await session.execute(query)
    invite_code = result.scalar_one_or_none()

    if not invite_code:
        raise ValueError("Invite code not found or does not belong to user")

    # Update fields if provided
    if update_request.state is not None:
        invite_code.state = SpecialInviteCodeState(update_request.state)
    if update_request.usage_limit is not None:
        invite_code.usage_limit = update_request.usage_limit

    await session.commit()
