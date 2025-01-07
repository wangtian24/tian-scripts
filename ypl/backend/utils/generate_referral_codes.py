from random_slugs.generate import generate_slug, get_total_unique_slugs
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from ypl.backend.db import get_async_engine
from ypl.db.invite_codes import SpecialInviteCode, SpecialInviteCodeState
from ypl.db.users import User

SLUG_OPTIONS = {
    "parts_of_speech": [
        "adjectives",
        "nouns",
        "nouns",
    ],
    "categories": {
        "adjectives": ["color", "quantity"],
        "nouns": ["animal", "pokemon", "technology", "transport", "profession"],
    },
    "format": "kebab",
}

NUMBER_OF_WORDS = 3


async def generate_invite_code_for_user(session: AsyncSession, user: User) -> SpecialInviteCode | None:
    # Check if user already has created invite codes
    query = select(SpecialInviteCode).where(
        SpecialInviteCode.creator_user_id == user.user_id, SpecialInviteCode.state == SpecialInviteCodeState.ACTIVE
    )
    existing_codes = (await session.exec(query)).all()

    if existing_codes:
        return None

    # Generate new invite code
    code = generate_slug(num_of_words=NUMBER_OF_WORDS, options=SLUG_OPTIONS)
    new_invite = SpecialInviteCode(code=code, creator_user_id=user.user_id, state=SpecialInviteCodeState.ACTIVE)
    session.add(new_invite)
    return new_invite


async def generate_invite_codes_for_yuppster_async() -> tuple[int, int]:
    """Generate invite codes for all users with emails ending in @yupp.ai.

    Returns:
        Tuple of (number of users found, number of new codes created)
    """
    async with AsyncSession(get_async_engine()) as session:
        print("entropy: ", get_total_unique_slugs(NUMBER_OF_WORDS, options=SLUG_OPTIONS))

        # Find all users with matching domain using string operations
        query = select(User).where(User.email.endswith("@yupp.ai"))
        users = (await session.exec(query)).all()

        codes_created = 0
        for user in users:
            code = await generate_invite_code_for_user(session, user)
            if code is not None:
                print(f"Generated code for user {user.email}: {code}")
                codes_created += 1

        await session.commit()
        return len(users), codes_created
