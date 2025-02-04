"""Backfill user profile fields from user_profiles table to users table.

This is a one-off script to move data from the user_profiles table to the users table
for the following fields:
- country_code (from country in user_profiles)
- city
- educational_institution
- discord_username

The data will only be copied if the corresponding field in the users table is null.
"""

from sqlalchemy import Connection, text
from sqlmodel import Session


def backfill_user_profile_fields(connection: Connection) -> None:
    """Backfill user profile fields from user_profiles to users table."""
    # Start a transaction
    with Session(connection) as session:
        # Update users table with values from user_profiles where the user fields are null
        # and the profile fields are not null
        session.execute(
            text(
                """
            UPDATE users u
            SET
                country_code = CASE
                    WHEN u.country_code IS NULL AND up.country IS NOT NULL
                    THEN up.country
                    ELSE u.country_code
                END,
                city = CASE
                    WHEN u.city IS NULL AND up.city IS NOT NULL
                    THEN up.city
                    ELSE u.city
                END,
                educational_institution = CASE
                    WHEN u.educational_institution IS NULL AND up.educational_institution IS NOT NULL
                    THEN up.educational_institution
                    ELSE u.educational_institution
                END,
                discord_username = CASE
                    WHEN u.discord_username IS NULL AND up.discord_username IS NOT NULL
                    THEN up.discord_username
                    ELSE u.discord_username
                END
            FROM user_profiles up
            WHERE u.user_id = up.user_id
            AND (
                (u.country_code IS NULL AND up.country IS NOT NULL)
                OR (u.city IS NULL AND up.city IS NOT NULL)
                OR (u.educational_institution IS NULL AND up.educational_institution IS NOT NULL)
                OR (u.discord_username IS NULL AND up.discord_username IS NOT NULL)
            )
            """
            )
        )
        session.commit()
