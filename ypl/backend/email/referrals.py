from sqlmodel import select
from ypl.backend.db import get_async_session
from ypl.backend.email.send_email import send_email_async
from ypl.db.users import User


async def send_referral_bonus_emails(
    new_user: str, new_user_credit_delta: int, referrer: str, referrer_credit_delta: int
) -> None:
    # Get both users' information
    new_user_model, referrer_model = await get_user_profiles(new_user, referrer)
    if not new_user_model:
        return

    # Send email to the new user if they got credits
    if new_user_credit_delta > 0 and new_user_model:
        await send_email_async(
            campaign="first_pref_bonus",
            to_address=new_user_model.email,
            template_params={
                "name": new_user_model.name,
                "credits": new_user_credit_delta,
            },
        )

    # Send email to the referrer if they got credits and we have their info
    if referrer_credit_delta > 0 and referrer_model and new_user_model:
        await send_email_async(
            campaign="referral_bonus",
            to_address=referrer_model.email,
            template_params={
                "referrer_name": referrer_model.name,
                "referee_name": new_user_model.name,
                "credits": referrer_credit_delta,
            },
        )


async def get_user_profiles(new_user_id: str, referrer_id: str) -> tuple[User | None, User | None]:
    """Get both users' information by their IDs.

    Args:
        new_user_id: The user_id of the new user
        referrer_id: The user_id of the referrer

    Returns:
        A tuple of (new_user, referrer_user) User objects, or None for any user not found
    """
    async with get_async_session() as session:
        query = select(User).where(User.user_id.in_([new_user_id, referrer_id]))  # type: ignore
        result = await session.exec(query)
        users = result.all()

        new_user_model = next((u for u in users if u.user_id == new_user_id), None)
        referrer_model = next((u for u in users if u.user_id == referrer_id), None)

        return (new_user_model, referrer_model)
