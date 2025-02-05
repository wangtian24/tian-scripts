from ypl.backend.email.campaigns.utils import load_html_template
from ypl.backend.email.email_types import EmailContent

SIC_AVAILABILITY_EMAIL_CONTENT = EmailContent(
    subject="Share Yupp with your friends!",
    preview=None,
    body_html=load_html_template("sic_availability_content.html"),
)

SIGN_UP_EMAIL_CONTENT = EmailContent(
    subject="Thanks for joining {brand_name} as an early Trusted Tester",
    preview=None,
    body_html=load_html_template("signup_email_content.html"),
)


# TODO: Add "Please continue to refer more of your friends by sharing <this URL with promo code> with them!"
YOUR_FRIEND_JOINED_EMAIL_CONTENT = EmailContent(
    subject="{referee_name} is now on {brand_name}",
    preview=None,
    body_html=load_html_template("your_friend_joined_content.html"),
)


# TODO: Add "Please continue to refer more of your friends by sharing <this URL with promo code> with them!"
REFFERAL_BONUS_EMAIL_CONTENT = EmailContent(
    subject="You just scored a referral bonus of {credits} YUPP credits!",
    preview=None,
    body_html=load_html_template("referral_bonus_content.html"),
)


FIRST_PREF_BONUS_EMAIL_CONTENT = EmailContent(
    subject="You just scored {credits} YUPP credits!",
    preview=None,
    body_html=load_html_template("first_pref_bonus_content.html"),
)
