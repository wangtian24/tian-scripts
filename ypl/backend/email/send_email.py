import logging
from typing import Any

import resend
from resend.emails._email import Email
from ypl.backend.config import settings
from ypl.backend.email.campaigns.nudge import (
    WEEK_1_CHECKIN_EMAIL_TEMPLATE,
    WEEK_1_CHECKIN_EMAIL_TITLE,
    WEEK_1_INACTIVE_EMAIL_TEMPLATE,
    WEEK_1_INACTIVE_EMAIL_TITLE,
    WEEK_5_INACTIVE_EMAIL_TEMPLATE,
    WEEK_5_INACTIVE_EMAIL_TITLE,
    WEEK_6_DEACTIVATION_EMAIL_TEMPLATE,
    WEEK_6_DEACTIVATION_EMAIL_TITLE,
)
from ypl.backend.email.campaigns.signup import (
    FIRST_PREF_BONUS_EMAIL_TEMPLATE,
    FIRST_PREF_BONUS_EMAIL_TITLE,
    REFFERAL_BONUS_EMAIL_TEMPLATE,
    REFFERAL_BONUS_EMAIL_TITLE,
    SIC_AVAILABILITY_EMAIL_TEMPLATE,
    SIC_AVAILABILITY_EMAIL_TEMPLATE_HTML,
    SIC_AVAILABILITY_EMAIL_TITLE,
    SIGN_UP_EMAIL_TEMPLATE,
    SIGN_UP_EMAIL_TITLE,
    YOUR_FRIEND_JOINED_EMAIL_TEMPLATE,
    YOUR_FRIEND_JOINED_EMAIL_TITLE,
)
from ypl.backend.utils.json import json_dumps

EMAIL_CAMPAIGNS = {
    "signup": {
        "title": SIGN_UP_EMAIL_TITLE,
        "template": SIGN_UP_EMAIL_TEMPLATE,
    },
    "sic_availability": {
        "title": SIC_AVAILABILITY_EMAIL_TITLE,
        "template": SIC_AVAILABILITY_EMAIL_TEMPLATE,
        "template_html": SIC_AVAILABILITY_EMAIL_TEMPLATE_HTML,
    },
    "referral_bonus": {
        "title": REFFERAL_BONUS_EMAIL_TITLE,
        "template": REFFERAL_BONUS_EMAIL_TEMPLATE,
    },
    "referred_user": {  # deprecated, use first_pref_bonus
        "title": FIRST_PREF_BONUS_EMAIL_TITLE,
        "template": FIRST_PREF_BONUS_EMAIL_TEMPLATE,
    },
    "first_pref_bonus": {
        "title": FIRST_PREF_BONUS_EMAIL_TITLE,
        "template": FIRST_PREF_BONUS_EMAIL_TEMPLATE,
    },
    "your_friend_joined": {
        "title": YOUR_FRIEND_JOINED_EMAIL_TITLE,
        "template": YOUR_FRIEND_JOINED_EMAIL_TEMPLATE,
    },
    "week_1_checkin": {
        "title": WEEK_1_CHECKIN_EMAIL_TITLE,
        "template": WEEK_1_CHECKIN_EMAIL_TEMPLATE,
    },
    "week_1_inactive": {
        "title": WEEK_1_INACTIVE_EMAIL_TITLE,
        "template": WEEK_1_INACTIVE_EMAIL_TEMPLATE,
    },
    "week_5_inactive": {
        "title": WEEK_5_INACTIVE_EMAIL_TITLE,
        "template": WEEK_5_INACTIVE_EMAIL_TEMPLATE,
    },
    "week_6_deactivation": {
        "title": WEEK_6_DEACTIVATION_EMAIL_TITLE,
        "template": WEEK_6_DEACTIVATION_EMAIL_TEMPLATE,
    },
}

REPLY_TO_ADDRESS = "gcmouli+yupp@yupp.ai"

BRAND_NAME = "Yupp (Alpha)"
CONFIDENTIALITY_FOOTER = """----
Thanks for being a part of our small, invite-only alpha.
We really appreciate your trust and ask for your strict confidentiality.
"""
INVITE_FRIEND_BONUS_CREDITS = "10,000"


async def send_email_async(campaign: str, to_address: str, template_params: dict[str, Any]) -> Email | None:
    if campaign not in EMAIL_CAMPAIGNS:
        raise ValueError(f"Campaign '{campaign}' not found")

    template_params = {
        **template_params,
        "BRAND_NAME": BRAND_NAME,
        "CONFIDENTIALITY_FOOTER": CONFIDENTIALITY_FOOTER,
        "credits": template_params.get("credits", INVITE_FRIEND_BONUS_CREDITS),
    }

    campaign_data = EMAIL_CAMPAIGNS[campaign]
    try:
        email_title = campaign_data["title"].format(**template_params)
        email_body = campaign_data["template"].format(**template_params)
        email_body_html = (
            campaign_data["template_html"].format(**template_params) if "template_html" in campaign_data else None
        )
    except KeyError as e:
        raise ValueError(f"Missing required parameter: {e}") from e

    resend_params: resend.Emails.SendParams = {
        # TODO(minqi): Finalize sender and reply-to address
        "from": "Mouli <mouli@updates.yupp.ai>",
        "to": [to_address],
        "reply_to": REPLY_TO_ADDRESS,
        "bcc": "system-email-log@yupp.ai",
        "subject": email_title,
        "text": email_body,
    }
    if email_body_html:
        resend_params["html"] = email_body_html

    logging.info(f"Email composed: {json_dumps(resend_params)}")

    if settings.resend_api_key:
        resend.api_key = settings.resend_api_key
        email = resend.Emails.send(resend_params)
        logging.info(f"Email sent: {json_dumps(email)}")
        return email
    else:
        return None
