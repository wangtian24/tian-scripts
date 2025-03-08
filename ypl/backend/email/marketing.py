import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, case, func
from sqlmodel import Session, select
from ypl.backend.email.send_email import EmailConfig, batch_send_emails_async
from ypl.db.chats import Chat, ChatMessage, Eval, EvalType, MessageEval
from ypl.db.emails import EmailLogs
from ypl.db.language_models import LanguageModel
from ypl.db.point_transactions import PointsActionEnum, PointTransaction
from ypl.db.users import User, UserStatus

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
    """Get users who were created within the given date range, along with their
    chat status, whether they unsubscribed, and whether they've received the
    campaign email.

    Args:
        start_date: Start of the date range (inclusive)
        end_date: End of the date range (inclusive)
        campaign: Campaign name to check for previous sends
    """
    has_chats = select(1).where(Chat.creator_user_id == User.user_id).exists().label("has_chats")
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
        User.deleted_at.is_(None),  # type: ignore
        User.status != UserStatus.DEACTIVATED,
        User.unsubscribed_from_marketing.is_(False),  # type: ignore
    ]

    return select(User, has_chats, already_sent).where(*conditions)


async def send_marketing_emails_async(
    session: Session,
    print_only: bool = False,
) -> None:
    """Send check-in emails to users based on their activity.

    Sends different templates in batches:
    - week_1_checkin: for active users who have completed evals (after 1 week)
    - week_1_inactive: for users who haven't completed any evals (after 1 week)
    - week_5_inactive: for users with no evals (after 5 weeks)
    - week_6_deactivation: for users with no evals (after 6 weeks)
        Note: Users receiving the week 6 deactivation email will have their accounts deactivated
    """
    all_email_configs = []
    users_to_deactivate = []

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
            for user, has_chats, already_sent in results:
                if already_sent or not has_chats:
                    if already_sent:
                        logging.info(f"Skipping {user.email} for {active_campaign} because email was already sent")
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
            for user, has_chats, already_sent in results:
                if already_sent or has_chats:
                    if already_sent:
                        logging.info(f"Skipping {user.email} for {inactive_campaign} because email was already sent")
                    continue

                if inactive_campaign == "week_6_deactivation":
                    users_to_deactivate.append(user)

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

    if print_only:
        logging.info(f"[PRINT-ONLY] Found {len(all_email_configs)} total emails to send:")
        for config in all_email_configs:
            logging.info(f"  - To: {config.to_address} | Campaign: {config.campaign}")
        if users_to_deactivate:
            logging.info(f"[PRINT-ONLY] Would deactivate {len(users_to_deactivate)} users")
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

        # After successfully sending emails, deactivate users
        if users_to_deactivate:
            logging.info(f"Deactivating {len(users_to_deactivate)} inactive users")
            for user in users_to_deactivate:
                user.status = UserStatus.DEACTIVATED
                logging.info(f"Deactivated user {user.email}")
            session.commit()

        logging.info("✓ All emails batched successfully")


def _users_eligible_for_summary() -> Any:
    """Get users who have at least 3 evals from the past month."""

    # Get evals from the past month
    one_month_ago = datetime.now() - timedelta(days=30)

    # Format campaign name for current month
    current_date = datetime.now(UTC)
    monthly_campaign_for_logging = (
        f"monthly_summary_{current_date.year}_{current_date.month:02d}_{current_date.day:02d}"
    )

    eval_count = (
        select(func.count())
        .where(
            and_(
                Eval.user_id == User.user_id,  # type: ignore
                Eval.eval_type == EvalType.SELECTION,  # type: ignore
                Eval.created_at >= one_month_ago,  # type: ignore
            )
        )
        .label("eval_count")
    )

    already_sent = (
        select(1)
        .where(
            EmailLogs.email_sent_to == User.email,
            EmailLogs.campaign_name == monthly_campaign_for_logging,
        )
        .exists()
        .label("already_sent")
    )

    favorite_model = (
        select(LanguageModel.label)
        .select_from(ChatMessage)
        .join(MessageEval, MessageEval.message_id == ChatMessage.message_id)  # type: ignore
        .join(
            Eval,
            and_(
                Eval.eval_id == MessageEval.eval_id,  # type: ignore
                Eval.user_id == User.user_id,  # type: ignore
                Eval.eval_type == EvalType.SELECTION,  # type: ignore
                Eval.created_at >= one_month_ago,  # type: ignore
            ),
        )
        .join(LanguageModel, LanguageModel.language_model_id == ChatMessage.assistant_language_model_id)  # type: ignore
        .where(MessageEval.score == 100)
        .group_by(LanguageModel.label)
        .order_by(func.count().desc())
        .limit(1)
        .scalar_subquery()
    ).label("favorite_model")

    credits_received = (
        select(func.sum(PointTransaction.point_delta))
        .where(
            PointTransaction.user_id == User.user_id,
            PointTransaction.action_type == PointsActionEnum.REWARD,
            PointTransaction.created_at >= one_month_ago,  # type: ignore
        )
        .label("credits_received")
    )

    credits_cashed_out = (
        select(
            func.sum(
                case(
                    (PointTransaction.action_type == PointsActionEnum.CASHOUT_REVERSED, -PointTransaction.point_delta),  # type: ignore
                    (PointTransaction.action_type == PointsActionEnum.CASHOUT, PointTransaction.point_delta),  # type: ignore
                    else_=0,
                )
            )
        )
        .where(
            PointTransaction.user_id == User.user_id,
            PointTransaction.created_at >= one_month_ago,  # type: ignore
            PointTransaction.action_type.in_([PointsActionEnum.CASHOUT, PointsActionEnum.CASHOUT_REVERSED]),  # type: ignore
        )
        .label("credits_cashed_out")
    )

    conditions = [
        User.deleted_at.is_(None),  # type: ignore
        User.status != UserStatus.DEACTIVATED,
        eval_count >= 3,
        # TODO(w): Enable this for external users once ready.
        User.email.endswith("yupp.ai"),
    ]

    return select(User, eval_count, already_sent, credits_received, credits_cashed_out, favorite_model).where(  # type: ignore
        *conditions
    )


async def send_monthly_summary_emails_async(session: Session, print_only: bool = False) -> None:
    # Format campaign name for current month
    current_date = datetime.now(UTC)
    monthly_campaign_for_logging = (
        f"monthly_summary_{current_date.year}_{current_date.month:02d}_{current_date.day:02d}"
    )

    all_email_configs = []
    results = session.exec(_users_eligible_for_summary()).all()
    for user, eval_count, already_sent, credits_received, credits_cashed_out, favorite_model in results:
        if already_sent:
            logging.info(f"Skipping {user.email} for {monthly_campaign_for_logging} because email was already sent")
            continue
        # Convert None to 0 and get absolute value of negative numbers
        credits_cashed_out_display = abs(credits_cashed_out) if credits_cashed_out is not None else 0
        # Format numbers with comma separators
        eval_count_fmt = f"{eval_count:,}"
        credits_received_fmt = f"{credits_received:,}" if credits_received is not None else "0"
        credits_cashed_out_fmt = f"{credits_cashed_out_display:,}"
        all_email_configs.append(
            EmailConfig(
                # "monthly_summary" is the campaign name we use for email
                # templates. On the EmailLogs table, we store the campaign name
                # as "monthly_summary_{year}_{month}_{day}".
                campaign="monthly_summary",
                to_address=user.email,
                template_params={
                    "email_recipient_name": user.name,
                    "pref_count": eval_count_fmt,
                    "credits_received": credits_received_fmt,
                    "credits_cashed_out": credits_cashed_out_fmt,
                    "favorite_model": favorite_model,
                },
            )
        )

    if print_only:
        logging.info(f"[PRINT-ONLY] Found {len(all_email_configs)} total emails to send:")
        for config in all_email_configs:
            logging.info(f"  - To: {config.to_address} | Campaign: {config.campaign}")
    elif all_email_configs:
        await batch_send_emails_async(all_email_configs)
