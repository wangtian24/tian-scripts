import os
import uuid

import psycopg2
from dotenv import load_dotenv

from ypl.db.users import WaitlistStatus

# This script is not run automatically with the migration cause it contains
# PIIs. If you need to backfill it locally, add the data here and run it
# manually.
# Run it with `python -m ypl.db.oneoffs.waitlisted_users`
EMAIL_TO_DISCORD_ID: dict[str, str] = {
    # "example@yupp.ai": "1234123412341234123",
    # Add more as needed, in the same format as the example above.
}

WAITLISTED_USER_INFO: dict[str, tuple[str, WaitlistStatus, str]] = {
    # "guest@gmail.com": ("yuppster@yupp.ai", WaitlistStatus.ALLOWED, "sibling who is in college"),
    # Add more as needed, in the same format as the example above.
}


def backfill_discord_ids_and_waitlist() -> None:
    """
    Backfills discord_ids in the users table and creates entries in the waitlisted_users table.
    Using direct database connection from .env credentials.
    """
    # Load environment variables
    load_dotenv()

    # Get database connection details from .env
    db_params = {
        "host": os.getenv("POSTGRES_HOST").split(":")[0],
        "port": os.getenv("POSTGRES_HOST").split(":")[1],
        "database": os.getenv("POSTGRES_DATABASE"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }

    # Connect to the database
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()

    try:
        # Update discord_ids for existing users
        for email, discord_id in EMAIL_TO_DISCORD_ID.items():
            if discord_id:
                cur.execute("UPDATE users SET discord_id = %s WHERE email = %s", (discord_id, email))

        # Create waitlisted users entries
        for email, (referrer_email, status, comment) in WAITLISTED_USER_INFO.items():
            # Look up the referrer's user_id
            referrer_id = None
            if referrer_email:
                cur.execute("SELECT user_id FROM users WHERE email = %s", (referrer_email,))
                result = cur.fetchone()
                referrer_id = result[0] if result else None

            # Convert WaitlistStatus enum to string
            status_str = status.value

            # Insert or update waitlisted user
            waitlisted_user_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO waitlisted_users (waitlisted_user_id, email, status, referrer_id, comment)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (email) DO UPDATE
                SET status = EXCLUDED.status,
                    referrer_id = EXCLUDED.referrer_id,
                    comment = EXCLUDED.comment
            """,
                (waitlisted_user_id, email, status_str, referrer_id, comment),
            )

        conn.commit()

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    backfill_discord_ids_and_waitlist()
