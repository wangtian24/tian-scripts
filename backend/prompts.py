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
