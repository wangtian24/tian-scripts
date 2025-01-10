import logging
from typing import Any

import resend
from resend.emails._email import Email
from ypl.backend.config import settings
from ypl.backend.email.campaigns.signup import (
    REFFERAL_BONUS_EMAIL_TEMPLATE,
    REFFERAL_BONUS_EMAIL_TITLE,
    REFFERED_USER_EMAIL_TEMPLATE,
    REFFERED_USER_EMAIL_TITLE,
    SIC_AVAILABILITY_EMAIL_TEMPLATE,
    SIC_AVAILABILITY_EMAIL_TEMPLATE_HTML,
    SIC_AVAILABILITY_EMAIL_TITLE,
    SIGN_UP_EMAIL_TEMPLATE,
    SIGN_UP_EMAIL_TITLE,
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
    "referred_user": {
        "title": REFFERED_USER_EMAIL_TITLE,
        "template": REFFERED_USER_EMAIL_TEMPLATE,
    },
}

REPLY_TO_ADDRESS = "gcmouli+yupp@yupp.ai"


async def send_email_async(campaign: str, to_address: str, params: dict[str, Any]) -> Email | None:
    if campaign not in EMAIL_CAMPAIGNS:
        raise ValueError(f"Campaign '{campaign}' not found")

    campaign_data = EMAIL_CAMPAIGNS[campaign]
    try:
        email_title = campaign_data["title"].format(**params)
        email_body = campaign_data["template"].format(**params)
        email_body_html = campaign_data["template_html"].format(**params) if "template_html" in campaign_data else None
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
