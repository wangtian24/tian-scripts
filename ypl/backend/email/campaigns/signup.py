from ypl.backend.email.campaigns.utils import load_html_template

################################################################################


SIC_AVAILABILITY_EMAIL_TITLE = "Share Yupp with your friends!"
SIC_AVAILABILITY_EMAIL_TEMPLATE = """
Hey {email_recipient_name},

We just gifted you with your own referral code!

Now you can invite your friends to try {BRAND_NAME}:
From your profile picture on the top right corner, click "Refer your friends" to see your special invite code.
Ask your friends to visit https://gg.yupp.ai, login using their Google account, and enter your invite code when prompted. Note: You can currently invite up to 3 friends per week.


For each friend you refer, you’ll get 10,000 YUPP credits and your friend will get 1,000. Please don’t refer to journalists, investors, competitors, etc, and continue to keep the product confidential. We want to remain in stealth for now.

Thanks for actively using Yupp, we hope you’re enjoying your experience and finding Yupp useful.

- Mouli, product team
"""  # noqa: E501

SIC_AVAILABILITY_EMAIL_TEMPLATE_HTML = """
<p>Hey {email_recipient_name},</p>

<p>We just gifted you with your own referral code!</p>

<p>Now you can invite your friends to try {BRAND_NAME}:</p>

<ol>
<li>From your profile picture on the top right corner, click "Refer your friends" to see your special invite code.</li>
<li>Ask your friends to visit <a href="https://gg.yupp.ai">https://gg.yupp.ai</a>, login using their Google account, and enter your invite code when prompted. <em>Note: You can currently invite up to 3 friends per week.</em></li>
</ol>

<p>For each friend you refer, you’ll get 10,000 YUPP credits and your friend will get 1,000. Please don’t refer to journalists, investors, competitors, etc, and continue to keep the product confidential. We want to remain in stealth for now.</p>

<p>Thanks for actively using Yupp, we hope you’re enjoying your experience and finding Yupp useful.</p>

- Mouli, product team
"""  # noqa: E501


################################################################################


SIGN_UP_EMAIL_TITLE = "Thanks for joining {BRAND_NAME} as an early Trusted Tester"
SIGN_UP_EMAIL_TEMPLATE = """
Hey {email_recipient_name},

We’re thrilled to have you here!

A few important points before you dive in:
• Confidential Access: This early access is exclusive to you, so please refrain from sharing it with others. Your discretion is greatly appreciated.
• Logging: To help us improve, we’ll be logging extensively. Please avoid writing any prompts that you consider confidential.
• We are still in alpha: That means you may encounter some rough edges or unexpected issues. We appreciate your patience as we work through it!

Try it out now – send a prompt to the AIs and pick the one you prefer. We will give you a one-time 1,000 YUPP credits as a treat.

For updates, AI discussions with our devs (and other Trusted Testers), and funny memes, join our Discord community here: https://discord.gg/AGHSbyqgXP

Your feedback plays a crucial role in refining the product, we can’t wait to hear your thoughts.

- Mouli, product team


{CONFIDENTIALITY_FOOTER}
"""  # noqa: E501

SIGN_UP_EMAIL_TEMPLATE_HTML = load_html_template("signup_email_content.html", SIGN_UP_EMAIL_TEMPLATE)

################################################################################
# TODO: Add "Please continue to refer more of your friends by sharing <this URL with promo code> with them!"


YOUR_FRIEND_JOINED_EMAIL_TITLE = "{referee_name} is now on {BRAND_NAME}"
YOUR_FRIEND_JOINED_EMAIL_TEMPLATE = """
Hey {email_recipient_name}:

{referee_name} just joined you on {BRAND_NAME} with your invite code.

Once they finish writing their first prompt and doing their first "I prefer this" feedback, you’ll be eligible to earn the referral bonus of {credits} YUPP credits.

- Mouli, product team

{CONFIDENTIALITY_FOOTER}
"""  # noqa: E501


################################################################################
# TODO: Add "Please continue to refer more of your friends by sharing <this URL with promo code> with them!"

REFFERAL_BONUS_EMAIL_TITLE = "You just scored a referral bonus of {credits} YUPP credits!"
REFFERAL_BONUS_EMAIL_TEMPLATE = """
Hey {email_recipient_name},

{referee_name} has signed up using your invite code and has started using Yupp–granting you a referral bonus of {credits} YUPP credits.

- Mouli, product team


{CONFIDENTIALITY_FOOTER}
"""  # noqa: E501


################################################################################


FIRST_PREF_BONUS_EMAIL_TITLE = "You just scored {credits} YUPP credits!"
FIRST_PREF_BONUS_EMAIL_TEMPLATE = """
Hey {email_recipient_name},

Thank you for giving your first feedback to the AI models. Since you joined Yupp with your friend’s invite code, you’ve been rewarded {credits} YUPP credits!

- Mouli, product team


{CONFIDENTIALITY_FOOTER}
"""  # noqa: E501
