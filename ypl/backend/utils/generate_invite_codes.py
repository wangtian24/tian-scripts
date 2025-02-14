import asyncio
import logging

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel.ext.asyncio.session import AsyncSession
from ypl.backend.config import settings
from ypl.backend.db import get_async_engine
from ypl.backend.llm.invite import get_users_eligible_for_invite_codes
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.soul_utils import get_soul_url
from ypl.db.invite_codes import SpecialInviteCode, SpecialInviteCodeState
from ypl.random_word_slugs.generate import Options, generate_slug

SLUG_OPTIONS: Options = {
    "format": "kebab",
}

NUMBER_OF_WORDS = 3

# Number of attempts allowed to try to generate a unique code for each user.
# Invite codes must be globally unique. Given our current user count, collision is very unlikely.
MAX_RETRIES = 3

logger = logging.getLogger()


async def generate_invite_code_for_top_users(
    min_age_days: int = 0,
    min_eval_days: int = 2,
    min_feedback_count: int = 0,
    limit: int = 10,
    default_active: bool = False,
) -> tuple[int, int]:
    """Generate invite codes for top users based on the given criteria."""
    async with AsyncSession(get_async_engine()) as session:
        users = await get_users_eligible_for_invite_codes(
            session,
            min_age_days=min_age_days,
            min_eval_days=min_eval_days,
            min_feedback_count=min_feedback_count,
            limit=limit,
        )
        codes_created = 0
        slack_messages = []

        for user in users:
            code = await generate_invite_code_for_user(session, user.user_id, default_active=default_active)
            if code is not None:
                generation_info = {
                    "user_id": user.user_id,
                    "name": user.name,
                    "code": code.code,
                    "usage_limit": code.usage_limit,
                    "state": code.state.value,
                }
                logging.info(json_dumps(generation_info))
                slack_messages.append(f"SIC for {user.name} {get_soul_url(user.user_id)} created: {code.code}")
                codes_created += 1

        await session.commit()

        if slack_messages:
            message_to_post = f"Generated {codes_created} SICs. <@U07BX3T7YBV> to review: \n\n" + "\n".join(
                slack_messages
            )

            await post_to_slack(
                message_to_post,
                webhook_url=settings.guest_management_slack_webhook_url,
            )

    return len(users), codes_created


async def generate_invite_code_for_user(
    session: AsyncSession, user_id: str, default_active: bool = False
) -> SpecialInviteCode | None:
    for attempt in range(MAX_RETRIES):
        try:
            # TODO(minqi): instead of retrying for up to 3 times,
            # one improvement is to just generate 2x number of codes and remove any entries that exists in the DB.

            # Create a savepoint before attempting to generate the code
            async with session.begin_nested():
                # Generate new invite code
                code = generate_slug(num_of_words=NUMBER_OF_WORDS, options=SLUG_OPTIONS)
                new_invite = SpecialInviteCode(
                    code=code,
                    creator_user_id=user_id,
                    state=SpecialInviteCodeState.ACTIVE if default_active else SpecialInviteCodeState.INACTIVE,
                    usage_limit=3,
                )
                session.add(new_invite)
                # Try to flush to catch any unique constraint violations, for the case that the code already exists
                await session.flush()
                return new_invite
        except SQLAlchemyError:
            if attempt == MAX_RETRIES - 1:  # Last attempt
                log_dict = {
                    "message": f":x: Failed to generate unique invite code after {MAX_RETRIES} attempts",
                    "user_id": user_id,
                    "soul_url": get_soul_url(user_id),
                }
                logging.warning(json_dumps(log_dict))
                asyncio.create_task(
                    post_to_slack(json_dumps(log_dict), webhook_url=settings.guest_management_slack_webhook_url)
                )
                return None
            # The savepoint will be automatically rolled back due to the exception
            continue

    return None
