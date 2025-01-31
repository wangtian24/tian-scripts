from sqlmodel import Session
from ypl.backend.email.send_email import send_email_async


async def send_marketing_emails_async(
    session: Session,
    dry_run: bool = False,
    limit: int | None = None,
) -> None:
    """Send emails for the specified campaign."""
    # TODO(w): Actual Implementation
    email_address = "delivered@resend.dev"
    if dry_run:
        print(f"Would send email to {email_address}:")
        print(f"Limit: {limit}")

    else:
        await send_email_async(
            # TODO(w): Change with actual campaign
            campaign="week_1_checkin",
            to_address=email_address,
            template_params={"email_recipient_name": "Test User"},
        )
