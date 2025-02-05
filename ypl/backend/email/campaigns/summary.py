from ypl.backend.email.campaigns.utils import load_html_template
from ypl.backend.email.email_types import EmailContent

MONTHLY_RECAP_EMAIL_CONTENT = EmailContent(
    subject="Your {brand_name} Monthly Recap",
    preview="Check out your {brand_name} activity from last month",
    body_html=load_html_template("monthly_summary.html"),
)
