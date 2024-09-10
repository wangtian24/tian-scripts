from langchain_core.prompts import ChatPromptTemplate

COMPARE_RESPONSES_SYSTEM_PROMPT = """
You are a specialized language model designed to analyze and compare responses from multiple LLMs to a given prompt.
Your task is to:

1. Take in a prompt and the responses from several LLMs.
2. Analyze these responses and generate a JSON output with the following keys:

  "summary": "A concise, one-sentence summary of the overall responses",
  "commonalities": "Brief description of shared elements across responses (1-2 sentences)",
  "differences": "Brief overview of unique aspects or divergences in each response (1-2 sentences)"

Ensure that each entry in the JSON is brief and to the point.
Focus on the most significant aspects of the responses in your analysis.

The input is structured as JSON too, with an entry for the prompt and another for each response,
keyed on the responding LLM.
"""

HIGHLIGHT_SIMILARITIES_SYSTEM_PROMPT = """
You are a specialized language model designed to analyze and compare responses from multiple LLMs to a given prompt.
Your task is to:

1. Take in a prompt and the responses from several LLMs.
2. Identify semantically similar sections in these responses.
3. Identify sections that are unique to each response.
4. Generate a JSON output with the following keys:

  "similar": "Tuples of sentences or text blocks that are similar across the responses.",
  "unique": "A list of sentences or text blocks that are unique to each response.",

Be sure to quote exact sentences from the responses in your output; do not modify them.

The input is structured as JSON too, with an entry for the prompt and another for each response,
keyed on the responding LLM.
"""


RESPONSES_USER_PROMPT = """

  "prompt": {prompt}

  "responses": {responses}

"""

PROMPT_DIFFICULTY_SYSTEM_PROMPT = """
You are a specialized language model designed to analyze prompts sent by users to LLMs.
Your task is to take in a prompt, and determine how complex it is for an LLM to answer it,
as a float between 0 and 1, where 0 is "easy" and 1 is "difficult".

Your response should be just the float, with no other text.
"""

PROMPT_DIFFICULTY_WITH_RESPONSES_SYSTEM_PROMPT = """
You are a specialized language model designed to analyze prompts sent by users to LLMs.
Your task is to take in a prompt and responses to it from different models,
and determine how complex it is for an LLM to answer it,
as a float between 0 and 1, where 0 is "easy" and 1 is "difficult".
One way to do this is to compare the responses; if they differ, the prompt is likely more complex.

Your response should be just the float, with no other text.
"""

PROMPT_DIFFICULTY_USER_PROMPT = "prompt: {prompt}"

PROMPT_CATEGORY_SYSTEM_TEMPLATE = """You are a specialized language model designed to analyze prompts sent by users to LLMs. Your task is to take in a prompt, and assign a category based on the prompt's content and intent, choosing from one of the following categories:

<REPLACE_WITH_CATEGORIES_FROM_DB>

Rules:
- Categorize based on the prompt's central topic and objective
- If the prompt could belong to more than one category, choose the one that best matches the primary objective of the prompt
"""

COMPARE_RESPONSES_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", COMPARE_RESPONSES_SYSTEM_PROMPT),
        ("user", RESPONSES_USER_PROMPT),
    ]
)

HIGHLIGHT_SIMILARITIES_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", HIGHLIGHT_SIMILARITIES_SYSTEM_PROMPT),
        ("user", RESPONSES_USER_PROMPT),
    ]
)

PROMPT_DIFFICULTY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_DIFFICULTY_SYSTEM_PROMPT),
        ("user", PROMPT_DIFFICULTY_USER_PROMPT),
    ]
)

PROMPT_DIFFICULTY_WITH_RESPONSES_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_DIFFICULTY_WITH_RESPONSES_SYSTEM_PROMPT),
        ("user", RESPONSES_USER_PROMPT),
    ]
)

SYNTHESIZER_FIRST_ASSISTANT_PROMPT = """Start chatting..."""

SYNTHESIZER_GENERATE_PERSONA_PROMPT = """Generate a JSON object representing persona, interests, and writing style, similar to the following examples:

{{"persona": "fantasy creature", "interests": ["mushrooms", "exploration", "meeting new beings"], "style": "playful informal"}}
{{"persona": "programmer", "interests": ["C++", "game development", "multithreading"], "style": "informal"}}
{{"persona": "scientist", "interests": ["physics", "mechanics", "experimentation"], "style": "formal"}}
{{"persona": "architectural historian", "interests": ["hypostyle hall design", "ancient architecture", "cross-cultural influences"], "style": "formal"}}
{{"persona": "artist", "interests": ["game design", "fantasy art", "mythology"], "style": "lowercase informal"}}
{{"persona": "fashion enthusiast", "interests": ["sustainable fashion", "upcycling", "sewing", "DIY repairs"], "style": "casual informal"}}
{{"persona": "gamer", "interests": ["SpongeBob", "Baldi's Basics", "game design"], "style": "casual"}}
{{"persona": "gamer", "interests": ["storytelling", "fantasy", "game mechanics"], "style": "lowercase informal"}}
{{"persona": "student", "interests": ["literature", "music", "media studies"], "style": "casual"}}
{{"persona": "content creator", "interests": ["social media", "challenges", "viral trends"], "style": "casual"}}
{{"persona": "linguist", "interests": ["german language", "grammar", "translation"], "style": "formal"}}
{{"persona": "scientist", "interests": ["immunology", "pathogen recognition", "cell biology"], "style": "formal"}}
{{"persona": "linguist", "interests": ["translation", "Japanese", "language learning"], "style": "formal"}}
{{"persona": "teenager", "interests": ["dragons", "friendship", "clubs"], "style": "casual"}}
{{"persona": "programmer", "interests": ["GPT 3.5", "Hugging Face", "Vercel", "app deployment"], "style": "informal"}}
{{"persona": "sports analyst", "interests": ["football", "statistics", "team performance"], "style": "informal"}}
{{"persona": "educator", "interests": ["learning techniques", "cognitive science", "personal development"], "style": "formal"}}
{{"persona": "historian", "interests": ["political systems", "ancient history", "sociopolitical dynamics"], "style": "formal"}}
{{"persona": "sports analyst", "interests": ["college football", "upset predictions", "team dynamics"], "style": "casual"}}
{{"persona": "fantasy writer", "interests": ["world-building", "character development", "mythology"], "style": "narrative descriptive"}}

Return {num_personas} JSON objects. Each MUST be on a separate new line. Do not explain or add markup."""

JUDGE_YUPP_CHAT_PROMPT = """User's prompt: {user_prompt}

Response 1: {response1}

(END RESPONSE 1)

Response 2: {response2}

(END RESPONSE 2)

Which of the above responses is better given the user's prompt? Say 1 if the first is much better, 2 if the first is slightly better, 3 if they are about the same, 4 if the second is slightly better, and 5 if the second is much better. Do not explain or add markup; only return the integer."""

WILDCHAT_REALISM_PROMPT = """Below are some WildChat prompts, followed by two of our prompts:

WildChat prompts:
{wildchat_prompt_str}

Our prompts:
Prompt 1: {prompt1}

Prompt 2: {prompt2}

Which of the two prompts is more similar to the WildChat prompts? Say 1 if the first is more similar and 2 if the second is more similar. Do not explain or add markup; only return the integer."""
