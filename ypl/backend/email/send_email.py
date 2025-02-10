import logging
from datetime import datetime
from typing import Any

import resend
from resend.emails._email import Email
from sqlalchemy import select
from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.email.campaigns.nudge import (
    WEEK_1_CHECKIN_EMAIL_CONTENT,
    WEEK_1_INACTIVE_EMAIL_CONTENT,
    WEEK_5_INACTIVE_EMAIL_CONTENT,
    WEEK_6_DEACTIVATION_EMAIL_CONTENT,
)
from ypl.backend.email.campaigns.signup import (
    FIRST_PREF_BONUS_EMAIL_CONTENT,
    REFFERAL_BONUS_EMAIL_CONTENT,
    SIC_AVAILABILITY_EMAIL_CONTENT,
    SIGN_UP_EMAIL_CONTENT,
    YOUR_FRIEND_JOINED_EMAIL_CONTENT,
)
from ypl.backend.email.campaigns.summary import MONTHLY_SUMMARY_EMAIL_CONTENT
from ypl.backend.email.campaigns.utils import html_to_plaintext, load_html_wrapper
from ypl.backend.email.email_types import EmailConfig, EmailContent
from ypl.backend.utils.json import json_dumps
from ypl.db.emails import EmailLogs
from ypl.db.users import User

EMAIL_CAMPAIGNS = {
    "signup": SIGN_UP_EMAIL_CONTENT,
    "sic_availability": SIC_AVAILABILITY_EMAIL_CONTENT,
    "referral_bonus": REFFERAL_BONUS_EMAIL_CONTENT,
    "referred_user": FIRST_PREF_BONUS_EMAIL_CONTENT,  # deprecated, use first_pref_bonus instead
    "first_pref_bonus": FIRST_PREF_BONUS_EMAIL_CONTENT,
    "your_friend_joined": YOUR_FRIEND_JOINED_EMAIL_CONTENT,
    "week_1_checkin": WEEK_1_CHECKIN_EMAIL_CONTENT,
    "week_1_inactive": WEEK_1_INACTIVE_EMAIL_CONTENT,
    "week_5_inactive": WEEK_5_INACTIVE_EMAIL_CONTENT,
    "week_6_deactivation": WEEK_6_DEACTIVATION_EMAIL_CONTENT,
    # "monthly_summary" is the campaign name we use for email
    # templates. On the EmailLogs table, we store the campaign name
    # as "monthly_summary_{year}_{month}_{day}".
    "monthly_summary": MONTHLY_SUMMARY_EMAIL_CONTENT,
}

REPLY_TO_ADDRESS = "gcmouli+yupp@yupp.ai"
BRAND_NAME = "Yupp (Alpha)"
SIGNATURE = """
<p>
  Mouli
  <br />
  Product Team
</p>
"""
INVITE_FRIEND_BONUS_CREDITS = "10,000"
CONFIDENTIALITY_FOOTER = """
<p>Thanks for being a part of our small, invite-only alpha.</p>
<p>We really appreciate your trust and ask for your strict confidentiality.</p>
"""
UNSUBSCRIBE_LINK = """
  <p>
    <a href="{unsubscribe_link}">Unsubscribe</a>
  </p>
"""


async def _get_user_id_from_email(email: str) -> str | None:
    """Get user ID from email address."""
    async with get_async_session() as session:
        result = await session.exec(select(User).where(User.email == email))  # type: ignore
        user = result.scalar_one_or_none()
        return user.user_id if user else None


async def _prepare_email_content(campaign: str, template_params: dict[str, Any]) -> EmailContent:
    """Prepare email content for a campaign.

    Returns:
        EmailContent object with subject, preview, body, and body_html
    """
    if campaign not in EMAIL_CAMPAIGNS:
        raise ValueError(f"Campaign '{campaign}' not found")

    template_params = {
        **template_params,
        "brand_name": BRAND_NAME,
        "confidentiality_footer": CONFIDENTIALITY_FOOTER,
        "signature": SIGNATURE,
        "credits": template_params.get("credits", INVITE_FRIEND_BONUS_CREDITS),
    }

    campaign_data = EMAIL_CAMPAIGNS[campaign]
    try:
        subject = campaign_data.subject.format(**template_params)
        preview_text = campaign_data.preview.format(**template_params) if campaign_data.preview else None
        body_html = (
            load_html_wrapper()
            .replace(
                "{{content}}", campaign_data.body_html.format(**template_params) if campaign_data.body_html else ""
            )
            .replace(
                "{{unsubscribe_link}}",
                UNSUBSCRIBE_LINK.format(**template_params) if "unsubscribe_link" in template_params else "",
            )
        )
        return EmailContent(
            subject=subject,
            preview=preview_text,
            body_html=body_html,
        )
    except KeyError as e:
        raise ValueError(f"Missing required parameter: {e}") from e


async def _log_emails_to_db(email_configs: list[EmailConfig]) -> None:
    """Log sent emails to the database.

    Args:
        email_configs: List of email configurations (campaign, to_address, template_params)
    """
    email_logs = [
        EmailLogs(
            email_sent_to=config.to_address,
            campaign_name=f"{config.campaign}_{datetime.now().strftime('%Y_%m:02d')}"
            if config.campaign == "monthly_summary"
            else config.campaign,
        )
        for config in email_configs
    ]
    try:
        async with get_async_session() as session:
            async with session.begin():
                session.add_all(email_logs)
                await session.commit()
    except Exception as e:
        logging.error(f"Failed to log emails to database: {str(e)}")


async def _create_email_params(
    to_address: str, email_title: str, email_body: str, email_body_html: str | None
) -> resend.Emails.SendParams:
    # TODO(w): pass this in as a param
    # user_id = await _get_user_id_from_email(to_address)

    """Create email parameters for Resend API."""
    params: resend.Emails.SendParams = {
        "from": "Mouli <mouli@updates.yupp.ai>",
        "to": [to_address],
        "reply_to": REPLY_TO_ADDRESS,
        "bcc": "system-email-log@yupp.ai",
        "subject": email_title,
        "text": email_body,
    }
    if email_body_html:
        params["html"] = email_body_html
    # TODO(w): Enable unsub link once UI is ready
    # if user_id:
    #     params["headers"] = {"List-Unsubscribe": f"https://gg.yupp.ai/unsubscribe/{user_id}"}
    return params


async def send_email_async(email_config: EmailConfig, print_only: bool = False) -> Email | None:
    """Send an email to a single recipient using the specified campaign template."""
    email_content = await _prepare_email_content(email_config.campaign, email_config.template_params)
    email_plaintext = html_to_plaintext(email_content.body_html)
    resend_params = await _create_email_params(
        email_config.to_address, email_content.subject, email_plaintext, email_content.body_html
    )

    if print_only:
        logging.info(f"Email composed: {json_dumps(resend_params)}")
        return None

    if settings.resend_api_key and not print_only:
        resend.api_key = settings.resend_api_key
        email = resend.Emails.send(resend_params)
        await _log_emails_to_db([email_config])
        logging.info(f"Email sent for {email_config.campaign}")
        return email
    return None


async def batch_send_emails_async(
    email_configs: list[EmailConfig],
) -> resend.Batch.SendResponse | None:
    """Send emails to multiple recipients with different campaigns in batch.

    Args:
        email_configs: List of email configurations (campaign, to_address, template_params)
    """
    batch_params = []

    # Prepare email parameters for each recipient
    for email_config in email_configs:
        email_content = await _prepare_email_content(email_config.campaign, email_config.template_params)
        email_plaintext = html_to_plaintext(email_content.body_html)
        batch_params.append(
            await _create_email_params(
                email_config.to_address,
                email_content.subject,
                email_plaintext,
                email_content.body_html,
            )
        )

    if settings.resend_api_key:
        resend.api_key = settings.resend_api_key
        response = resend.Batch.send(batch_params)
        await _log_emails_to_db(email_configs)
        logging.info(f"Email count sent in batch: {len(email_configs)}")
        return response
    return None
