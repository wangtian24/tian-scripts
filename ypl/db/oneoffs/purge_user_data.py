"""
This script hard-deletes users and all related records (chats, turns, evals, etc)
from the database, based on email addresses.
It is ordered based on the existing foreign key relationships as on date - 5-Dec-2024.
It needs to be updated for newer relationships.

Usage:
    python -m ypl.db.oneoffs.purge_user_data --emails "user1@example.com, user2@example.com"

Options:
    --emails    Comma-separated list of email addresses to delete.

Note:
    This will hard delete data, ensure the change is reviewed and approved by team.
"""

import argparse

from sqlmodel import Session, delete, select

from ypl.backend.db import get_engine
from ypl.db.chats import Chat, ChatMessage, Eval, MessageEval, PromptModifierAssoc, Turn, TurnQuality
from ypl.db.point_transactions import PointTransaction
from ypl.db.rewards import Reward, RewardActionLog
from ypl.db.users import Account, SyntheticUserAttributes, User, VerificationToken
from ypl.db.users import Session as UserSession


def delete_user_cascade(session: Session, user_id: str) -> None:
    """Delete a user and all related records in the correct order"""
    try:
        # 1. Delete sessions
        session.exec(delete(UserSession).where(UserSession.user_id == user_id))

        # 2. Delete accounts
        session.exec(delete(Account).where(Account.user_id == user_id))

        # 3. Delete verification tokens if any exist
        session.exec(delete(VerificationToken).where(VerificationToken.identifier == user_id))

        # 4. Delete synthetic user attributes
        session.exec(delete(SyntheticUserAttributes).where(SyntheticUserAttributes.user_id == user_id))

        # 5. Delete point transactions
        session.exec(delete(PointTransaction).where(PointTransaction.user_id == user_id))

        # 6. Delete reward action logs
        session.exec(delete(RewardActionLog).where(RewardActionLog.user_id == user_id))

        # 7. Delete rewards
        session.exec(delete(Reward).where(Reward.user_id == user_id))

        # 8. First get all eval IDs for the user
        eval_query = select(Eval.eval_id).where(Eval.user_id == user_id)
        eval_ids = session.exec(eval_query).all()

        # 9. Delete message_evals for these evals
        if eval_ids:
            session.exec(delete(MessageEval).where(MessageEval.eval_id.in_(eval_ids)))

        # 10. Then delete evals
        session.exec(delete(Eval).where(Eval.user_id == user_id))

        # 11. Delete chat messages and turns
        # First get all turns created by the user
        turns_query = select(Turn.turn_id).where(Turn.creator_user_id == user_id)
        turn_ids = session.exec(turns_query).all()

        if turn_ids:
            # Get all chat message IDs for these turns
            messages_query = select(ChatMessage.message_id).where(ChatMessage.turn_id.in_(turn_ids))
            message_ids = session.exec(messages_query).all()

            if message_ids:
                # Delete prompt modifier associations first
                session.exec(delete(PromptModifierAssoc).where(PromptModifierAssoc.chat_message_id.in_(message_ids)))

            # Delete associated chat messages
            session.exec(delete(ChatMessage).where(ChatMessage.turn_id.in_(turn_ids)))

            # Delete turn qualities associated with these turns
            session.exec(delete(TurnQuality).where(TurnQuality.turn_id.in_(turn_ids)))

            # Delete the turns
            session.exec(delete(Turn).where(Turn.turn_id.in_(turn_ids)))

        # 12. Delete chats created by the user
        session.exec(delete(Chat).where(Chat.creator_user_id == user_id))

        # 13. Finally delete the user
        session.exec(delete(User).where(User.user_id == user_id))

        session.commit()

    except Exception as e:
        session.rollback()
        raise Exception(f"Failed to delete user and related records: {str(e)}") from e


def delete_users_by_emails(email_list: str):
    """Delete users by their email addresses
    Args:
        email_list: Comma-separated string of email addresses
    """
    # Clean and parse email list
    emails = [email.strip() for email in email_list.split(",")]

    with Session(get_engine()) as session:
        try:
            # Fetch user_ids for the given emails
            query = select(User.user_id, User.email).where(User.email.in_(emails))
            result = session.exec(query)
            users = result.all()

            if not users:
                print("No users found for the provided email addresses")
                return

            # Process each user
            for user_id, email in users:
                try:
                    print(f"Deleting user {email} (ID: {user_id})...")
                    delete_user_cascade(session, user_id)
                    print(f"Successfully deleted user {email}")
                except Exception as e:
                    print(f"Failed to delete user {email}: {str(e)}")

            print(f"\nProcessed {len(users)} users")
            print(f"Found users for emails: {[user[1] for user in users]}")
            not_found = set(emails) - set(user[1] for user in users)
            if not_found:
                print(f"No users found for emails: {list(not_found)}")

        except Exception as e:
            print(f"Error processing users: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Delete users by email addresses")
    parser.add_argument("--emails", type=str, required=True, help="Comma-separated list of email addresses to delete")

    args = parser.parse_args()

    # Show warning and confirmation prompt
    emails = [email.strip() for email in args.emails.split(",")]
    print("\nWARNING: You are about to delete User & related Data:")
    for email in emails:
        print(f"  - {email}")

    confirmation = input("\nAre you sure you want to proceed? (y/n): ")
    if confirmation.lower() != "y":
        print("Operation cancelled.")
        return

    # Run the deletion
    delete_users_by_emails(args.emails)


if __name__ == "__main__":
    main()
