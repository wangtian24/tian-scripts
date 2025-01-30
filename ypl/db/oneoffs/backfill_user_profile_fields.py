"""Backfill user profile fields from user_profiles table to users table.

This is a one-off script to move data from the user_profiles table to the users table
for the following fields:
- country_code (from country in user_profiles)
- city
- educational_institution
- discord_username

The data will only be copied if the corresponding field in the users table is null.
"""

from sqlalchemy import Connection, select, update
from sqlmodel import Session

from ypl.db.users import User, UserProfile


def backfill_user_profile_fields(connection: Connection) -> None:
    """Backfill user profile fields from user_profiles to users table."""
    with Session(connection) as session:
        # Get all user profiles that have at least one of the fields we want to copy
        stmt = select(UserProfile).where(
            (UserProfile.country.is_not(None))
            | (UserProfile.city.is_not(None))
            | (UserProfile.educational_institution.is_not(None))
            | (UserProfile.discord_username.is_not(None))
        )
        result = session.execute(stmt)
        profiles = result.scalars().all()

        for profile in profiles:
            # For each profile, update the corresponding user only if the user's fields are null
            # and the profile has non-null values
            update_values = {}

            if profile.country is not None:
                update_values["country_code"] = profile.country

            if profile.city is not None:
                update_values["city"] = profile.city

            if profile.educational_institution is not None:
                update_values["educational_institution"] = profile.educational_institution

            if profile.discord_username is not None:
                update_values["discord_username"] = profile.discord_username

            if update_values:
                # Only update if there are values to update
                session.execute(
                    update(User)
                    .where(
                        User.user_id == profile.user_id,
                        # Only update if the all corresponding field in users table is null
                        User.country_code.is_(None),
                        User.city.is_(None),
                        User.educational_institution.is_(None),
                        User.discord_username.is_(None),
                    )
                    .values(**update_values)
                )

        session.commit()
