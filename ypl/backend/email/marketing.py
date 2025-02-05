import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlmodel import Session, select
from ypl.backend.email.send_email import EmailConfig, batch_send_emails_async
from ypl.db.chats import Eval
from ypl.db.emails import EmailLogs
from ypl.db.users import User

# In case cron job fails, we retry again every day for RETRY_DAYS days.
RETRY_DAYS = 3
RESEND_BATCH_SIZE = 100


@dataclass
class Campaign:
    active: str | None
    inactive: str | None


@dataclass
class TimeframeConfig:
    days: int
    campaign: Campaign


MARKETING_EMAIL_TIMEFRAMES: list[TimeframeConfig] = [
    TimeframeConfig(
        days=7,
        campaign=Campaign(active="week_1_checkin", inactive="week_1_inactive"),
    ),
    TimeframeConfig(
        days=35,
        campaign=Campaign(active=None, inactive="week_5_inactive"),
    ),
    TimeframeConfig(
        days=42,
        campaign=Campaign(active=None, inactive="week_6_deactivation"),
    ),
]


def _users_for_timeframe_query(
    start_date: datetime,
    end_date: datetime,
    campaign: str,
) -> Any:
    """Get users who were created within the given date range, along with their eval status
    and whether they've received the campaign email.

    Args:
        start_date: Start of the date range (inclusive)
        end_date: End of the date range (inclusive)
        campaign: Campaign name to check for previous sends
    """
    has_evals = select(1).where(Eval.user_id == User.user_id).exists().label("has_evals")
    already_sent = (
        select(1)
        .where(
            EmailLogs.email_sent_to == User.email,
            EmailLogs.campaign_name == campaign,
        )
        .exists()
        .label("already_sent")
    )

    conditions = [
        User.created_at.is_not(None),  # type: ignore
        User.created_at >= start_date,  # type: ignore
        User.created_at <= end_date,  # type: ignore
    ]

    return select(User, has_evals, already_sent).where(*conditions)


async def send_marketing_emails_async(
    session: Session,
    dry_run: bool = False,
) -> None:
    """Send check-in emails to users based on their activity.

    Sends different templates in batches:
    - week_1_checkin: for active users who have completed evals (after 1 week)
    - week_1_inactive: for users who haven't completed any evals (after 1 week)
    - week_5_inactive: for users with no evals (after 5 weeks)
    - week_6_deactivation: for users with no evals (after 6 weeks)
    """

    all_email_configs = []
    for timeframe in MARKETING_EMAIL_TIMEFRAMES:
        now = datetime.now(UTC)
        time_ago_start = now - timedelta(days=timeframe.days + RETRY_DAYS)
        time_ago_end = now - timedelta(days=timeframe.days)

        start_date = datetime(time_ago_start.year, time_ago_start.month, time_ago_start.day, tzinfo=UTC)
        end_date = datetime(time_ago_end.year, time_ago_end.month, time_ago_end.day, 23, 59, 59, tzinfo=UTC)

        active_campaign = timeframe.campaign.active
        inactive_campaign = timeframe.campaign.inactive

        # Query once for active campaign
        if active_campaign:
            query = _users_for_timeframe_query(start_date, end_date, active_campaign)
            results = session.exec(query).all()
            for user, has_evals, already_sent in results:
                if already_sent or not has_evals:
                    continue
                all_email_configs.append(
                    EmailConfig(
                        campaign=active_campaign,
                        to_address=user.email,
                        template_params={
                            "email_recipient_name": user.name,
                            # TODO(w): Enable unsub link once UI is ready
                            # "unsubscribe_link": f"https://gg.yupp.ai/unsubscribe?user_id={user.user_id}",
                        },
                    )
                )

        # Query once for inactive campaign
        if inactive_campaign:
            query = _users_for_timeframe_query(start_date, end_date, inactive_campaign)
            results = session.exec(query).all()
            for user, has_evals, already_sent in results:
                if already_sent or has_evals:
                    continue
                all_email_configs.append(
                    EmailConfig(
                        campaign=inactive_campaign,
                        to_address=user.email,
                        template_params={
                            "email_recipient_name": user.name,
                            # TODO(w): Enable unsub link once UI is ready
                            # "unsubscribe_link": f"https://gg.yupp.ai/unsubscribe?user_id={user.user_id}",
                        },
                    )
                )

    if dry_run:
        logging.info(f"[DRY RUN] Found {len(all_email_configs)} total emails to send:")
        for config in all_email_configs:
            logging.info(f"  - To: {config.to_address} | Campaign: {config.campaign}")
    elif all_email_configs:
        # Send emails in batches of RESEND_BATCH_SIZE
        for i in range(0, len(all_email_configs), RESEND_BATCH_SIZE):
            batch = all_email_configs[i : i + RESEND_BATCH_SIZE]
            logging.info(
                f"Batching {len(batch)} emails"
                f"({i + 1}-{min(i + RESEND_BATCH_SIZE, len(all_email_configs))} of {len(all_email_configs)})"
            )
            try:
                await batch_send_emails_async(batch)
            except Exception as e:
                logging.error(f"Error sending batch emails {i + 1}: {str(e)}")
                raise
        logging.info("✓ All emails batched successfully")
