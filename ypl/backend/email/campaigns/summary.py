MONTHLY_RECAP_EMAIL_TITLE = "Your {BRAND_NAME} Monthly Recap"
MONTHLY_RECAP_EMAIL_PREVIEW = "Check out your {BRAND_NAME} activity from last month"
MONTHLY_RECAP_EMAIL_TEMPLATE = """
Hey {email_recipient_name},

Recap time—a quick snapshot of how you've been using {BRAND_NAME}. Let's dive in:

✨ Your Top 3 Prompts

{prompt_1}
{prompt_2}
{prompt_3}

✨ Feedback Frenzy

You gave {pref_count} feedback on AI models—seems like you've got a good eye for quality responses.

✨ Favorite Model

Your go-to AI this month was {model_name}. Looks like you two really vibe!

✨ Credits Update

Received: {credits_received} credits
Cashed out: {credits_spent} credits

Keep up the great work, and here's to another month of smarter chats and better AIs. Let us know if you have any questions or feedback!


{SIGNATURE}

{CONFIDENTIALITY_FOOTER}
"""  # noqa: E501

MONTHLY_RECAP_EMAIL_TEMPLATE_HTML = """
<p>Hey {email_recipient_name},</p>

<p>Recap time—a quick snapshot of how you've been using {BRAND_NAME}. Let's dive in:</p>

<p>✨ Your Top 3 Prompts</p>

<p>{prompt_1}<br>
{prompt_2}<br>
{prompt_3}</p>

<p>✨ Feedback Frenzy</p>

<p>You gave {pref_count} feedback on AI models—seems like you've got a good eye for quality responses.</p>

<p>✨ Favorite Model</p>

<p>Your go-to AI this month was {model_name}. Looks like you two really vibe!</p>

<p>✨ Credits Update</p>

<p>Received: {credits_received} credits<br>
Cashed out: {credits_spent} credits</p>

<p>Keep up the great work, and here's to another month of smarter chats and better AIs. Let us know if you have any questions or feedback!</p>

<p>{SIGNATURE}</p>

{CONFIDENTIALITY_FOOTER}
"""  # noqa: E501


################################################################################
