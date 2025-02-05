from ypl.backend.email.campaigns.utils import html_to_plaintext, load_html_template
from ypl.backend.email.email_types import EmailContent

WEEK_1_CHECKIN_HTML = load_html_template("week_1_checkin_content.html")
WEEK_1_CHECKIN_EMAIL_CONTENT = EmailContent(
    subject="A quick check-in and important reminder",
    preview="Tell us your experience so far and please keep things under wraps",
    body_html=WEEK_1_CHECKIN_HTML,
    body=html_to_plaintext(WEEK_1_CHECKIN_HTML),
)

WEEK_1_INACTIVE_HTML = load_html_template("week_1_inactive_content.html")
WEEK_1_INACTIVE_EMAIL_CONTENT = EmailContent(
    subject="It's been a minute",
    preview="We noticed you haven't tried {brand_name} yet. Let's get you started!",
    body_html=WEEK_1_INACTIVE_HTML,
    body=html_to_plaintext(WEEK_1_INACTIVE_HTML),
)


WEEK_5_INACTIVE_HTML = load_html_template("week_5_inactive_content.html")
WEEK_5_INACTIVE_EMAIL_CONTENT = EmailContent(
    subject="Are you still there?",
    preview="Stop your account from being deactivated",
    body_html=WEEK_5_INACTIVE_HTML,
    body=html_to_plaintext(WEEK_5_INACTIVE_HTML),
)


WEEK_6_DEACTIVATION_HTML = load_html_template("week_6_deactivation_content.html")
WEEK_6_DEACTIVATION_EMAIL_CONTENT = EmailContent(
    subject="Your {brand_name} account has been deactivated",
    preview="Due to inactivity, we've deactivated your account—but you can always come back!",
    body_html=WEEK_6_DEACTIVATION_HTML,
    body=html_to_plaintext(WEEK_6_DEACTIVATION_HTML),
)
