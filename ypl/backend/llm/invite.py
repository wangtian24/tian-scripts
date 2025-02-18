import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import String, cast, desc, exists, func, not_
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ypl.backend.email.send_email import EmailConfig, send_email_async
from ypl.db.app_feedback import AppFeedback
from ypl.db.chats import Eval
from ypl.db.invite_codes import (
    SpecialInviteCode,
    SpecialInviteCodeClaimLog,
    SpecialInviteCodeState,
)
from ypl.db.users import User, UserStatus
from ypl.random_word_slugs.generate import Options, generate_slug


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


class EligibleUser(BaseModel):
    user_id: str
    name: str
    created_at: datetime
    num_eval_days: int  # Distinct days that the user has provided evals on.
    total_evals: int  # Total number of evals created
    feedback_count: int


class EligibleUsersResponse(BaseModel):
    users: list[EligibleUser]


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


async def get_users_eligible_for_invite_codes(
    session: AsyncSession,
    min_age_days: int,
    min_eval_days: int,
    min_feedback_count: int,
    limit: int = 10,
) -> list[EligibleUser]:
    """Get users eligible for invite codes based on a few criteria, ordered by decreasing activity."""

    cutoff_date = datetime.utcnow() - timedelta(days=min_age_days)

    # Get number of evals from this user.
    eval_stats = (
        select(
            cast(Eval.user_id, String).label("user_id"),
            func.count(func.distinct(func.date(Eval.created_at))).label("num_eval_days"),
            func.count().label("total_evals"),
        )
        .where(
            Eval.deleted_at.is_(None),  # type: ignore
        )
        .group_by(cast(Eval.user_id, String))
        .having(func.count(func.distinct(func.date(Eval.created_at))) >= min_eval_days)
        .cte("eval_stats")
    )

    # Get number of in-app feeedback sent by the user.
    feedback_stats = (
        select(cast(AppFeedback.user_id, String).label("user_id"), func.count().label("feedback_count"))
        .where(AppFeedback.deleted_at.is_(None))  # type: ignore
        .group_by(cast(AppFeedback.user_id, String))
        .cte("feedback_stats")
    )

    query = (
        select(
            User,
            eval_stats.c.num_eval_days,
            eval_stats.c.total_evals,
            func.coalesce(feedback_stats.c.feedback_count, 0).label("feedback_count"),
        )
        .select_from(User)
        .join(eval_stats, eval_stats.c.user_id == User.user_id)
        # Use left outer join since feedback_min_count is allowed to be 0.
        # Make sure we find users that don't have any in-app feedback.
        .outerjoin(feedback_stats, feedback_stats.c.user_id == User.user_id)
        .where(
            User.created_at <= cutoff_date,  # type: ignore
            User.deleted_at.is_(None),  # type: ignore
            User.status == UserStatus.ACTIVE,
            func.coalesce(feedback_stats.c.feedback_count, 0) >= min_feedback_count,
            # Exclude user that already have invite codes (both inactive or active)
            not_(
                exists(
                    select(1).where(
                        SpecialInviteCode.creator_user_id == User.user_id,
                        SpecialInviteCode.deleted_at.is_(None),  # type: ignore
                    )
                )
            ),
        )
        .order_by(
            eval_stats.c.num_eval_days.desc(),
            eval_stats.c.total_evals.desc(),
            User.created_at.desc(),  # type: ignore
        )
        .limit(limit)
    )

    result = await session.execute(query)
    users_with_evals = result.all()

    eligible_users = [
        EligibleUser(
            user_id=row.User.user_id,
            name=row.User.name,
            created_at=row.User.created_at,
            num_eval_days=row.num_eval_days,
            total_evals=row.total_evals,
            feedback_count=row.feedback_count,
        )
        for row in users_with_evals
    ]
    return eligible_users


SLUG_OPTIONS: Options = {
    "format": "kebab",
}


async def get_new_invite_code(session: AsyncSession) -> str:
    """Suggests a new invite code using word slugs without recording it in the database."""
    retry_remaining = 3
    while retry_remaining > 0:
        code = generate_slug(num_of_words=3, options=SLUG_OPTIONS)

        # Check if code already exists
        query = select(SpecialInviteCode).where(SpecialInviteCode.code == code)
        result = await session.execute(query)
        if result.first() is None:
            return code
        retry_remaining -= 1
    raise ValueError("Retry limit reached. Failed to generate a unique invite code after 3 attempts.")


async def create_invite_code_for_user(
    code: str,
    user_id: str,
    session: AsyncSession,
    usage_limit: int | None = None,
    referral_bonus_eligible: bool = True,
) -> UUID:
    """Create a new invite code with the specified code string and creator user ID.

    Args:
        code: The invite code string
        user_id: The ID of the user creating the invite code
        session: Database session
        usage_limit: Optional limit on number of times this code can be used
        referral_bonus_eligible: Whether users who use this code are eligible for referral bonus

    Returns:
        The UUID of the created invite code

    Raises:
        IntegrityError: If the invite code already exists (unique constraint violation)
        ValueError: If user already has active invite codes
    """
    # Query number of active invite codes for this user
    existing_codes_query = (
        select(func.count())
        .select_from(SpecialInviteCode)
        .where(
            SpecialInviteCode.creator_user_id == user_id,
            SpecialInviteCode.state == SpecialInviteCodeState.ACTIVE,
            SpecialInviteCode.deleted_at.is_(None),  # type: ignore
        )
    )
    result = await session.execute(existing_codes_query)
    existing_codes_count = result.scalar_one()

    invite_code = SpecialInviteCode(
        code=code,
        creator_user_id=user_id,
        usage_limit=usage_limit,
        referral_bonus_eligible=referral_bonus_eligible,
        state=SpecialInviteCodeState.ACTIVE,
    )
    session.add(invite_code)
    await session.commit()

    if existing_codes_count == 0:
        asyncio.create_task(send_sic_availability_email(session, user_id))

    return invite_code.special_invite_code_id


async def send_sic_availability_email(session: AsyncSession, user_id: str) -> None:
    # Send the sic_availability email for users getting an invite code for the first time.
    user_query = select(User).where(User.user_id == user_id)
    user_result = await session.execute(user_query)
    user = user_result.scalar_one_or_none()
    if user:
        await send_email_async(
            EmailConfig(
                campaign="sic_availability",
                to_address=user.email,
                template_params={
                    "email_recipient_name": user.name,
                },
            )
        )
