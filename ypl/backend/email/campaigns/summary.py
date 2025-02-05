from ypl.backend.email.campaigns.utils import html_to_plaintext, load_html_template
from ypl.backend.email.email_types import EmailContent

MONTHLY_RECAP_EMAIL_TEMPLATE_HTML = load_html_template("monthly_summary.html")
MONTHLY_RECAP_EMAIL_CONTENT = EmailContent(
    subject="Your {brand_name} Monthly Recap",
    preview="Check out your {brand_name} activity from last month",
    body_html=MONTHLY_RECAP_EMAIL_TEMPLATE_HTML,
    body=html_to_plaintext(MONTHLY_RECAP_EMAIL_TEMPLATE_HTML),
)
