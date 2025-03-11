from langchain_core.prompts import ChatPromptTemplate

JUDGE_SUGGESTED_FOLLOWUPS_PROMPT = """
You are a helpful assistant specializing in generating follow-up questions to conversations between users and LLMs.
Below is the conversation history between a user and an LLM.
Based on the the discussion, suggest up to 4 thoughtful follow-up questions the user can ask to deepen their understanding,
explore related ideas, or address unresolved aspects of the conversation.
Make the suggestions short, specific, directly relevant to the provided conversation, and different from one another.
The suggestions should be in the same language as the user's prompt.
Focus on the most recent user prompt in the conversation.
Return the list of follow-up questions as a JSON array, where each item contains the suggestion and a short 2-5 word label for it.

Output format:
[{{"suggestion": "...", "label": "..."}}, {{"suggestion": "...", "label": "..."}}]

Do not explain; only return the list as JSON. Do not include "```json" or "```" in your response.
Use the same language as the user's prompt.

Conversation history:
{chat_history}
"""

JUDGE_SUGGESTED_FOLLOWUPS_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [("human", JUDGE_SUGGESTED_FOLLOWUPS_PROMPT)]
)

JUDGE_SUGGESTED_PROMPTBOX_PROMPT = """
You are a helpful assistant specializing in generating placeholder text.
Below is the conversation history between a user and an LLM.
Based on the the discussion, suggest a placeholder text for a box used by the user to follow up on the conversation.
The placeholder text should broadly be related to the conversation, but not be a specific follow-up question;
instead, it should provide a generic follow-up encouragement related to the conversation.
Do not explain; only return the placeholder text as a string. Do not include markup in your response.
The placeholder text should be brief, ideally 10 words or less, and phrased as a suggestion to the user.

For example, if the conversation is about career opportunities, the placeholder text could be "Ask a follow-up career question"
If the user has asked about antibiotics, the placeholder text could be "Anything else you'd like to know about antibiotics?"
And so on. Keep it very short and concise.

Conversation history:
{chat_history}
"""

JUDGE_SUGGESTED_PROMPTBOX_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [("human", JUDGE_SUGGESTED_PROMPTBOX_PROMPT)]
)

JUDGE_CONVERSATION_STARTERS_PROMPT = """
You are an assistant skilled at creating engaging conversation starters between a user and an LLM.
Below are recent conversations a user has had with different LLMs.
Based on these, generate up to 10 engaging conversation starters for new chats this user may want to initiate with an LLM.
These conversation starters are intended to be used by the user to ask the LLM, not the LLM to ask the user.
They should explore related areas, or introduce fresh angles while staying relevant to the user's interests and prior discussions.
The conversation starters should not directly continue a previous conversation, but rather introduce a new, potentially
controversial or interesting topic, in an area that the user is interested in, based on the conversation history.
Give more weight to later conversations, but try to generate a diverse set of conversation starters.
Return the list of follow-up questions as a JSON array, where each item contains:
- the suggestion,
- a short 2-5 word label for it in title case - make sure to correctly capitalize any proper nouns and abbreviations such as "AI" or "US",
- an explanation of why it was selected, referring to the conversation history elements used.

Output format:
[{{"suggestion": "...", "label": "...", "explanation": "..."}}, {{"suggestion": "...", "label": "...", "explanation": "..."}}]

Do not explain; only return the list as JSON. Do not include "```json" or "```" in your response.

Conversation history:
{chat_history}
"""

JUDGE_CONVERSATION_STARTERS_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [("human", JUDGE_CONVERSATION_STARTERS_PROMPT)]
)


JUDGE_CHAT_TITLE_PROMPT = """
You are an assistant skilled at creating titles of a conversation between a user and one or more LLMs.
Below is such a conversation; based on this conversation, generate a brief title for the chat.
The title should be:
- Brief, ideally less than 6 words
- Informative, capturing the essence of the conversation
- In the same language as the user's prompt
Do not prefix the title or include any formatting - just return the title directly.

Conversation:
{chat_history}
"""

JUDGE_CHAT_TITLE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([("human", JUDGE_CHAT_TITLE_PROMPT)])
