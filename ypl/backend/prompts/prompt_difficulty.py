from langchain_core.prompts import ChatPromptTemplate

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

JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT = """
You are an AI assistant specialized in evaluating the difficulty of prompts given to language models.
When presented with a prompt and an optional response (or two responses from different models),
analyze them and rate the prompt's difficulty on a scale of 1 to 10,
where 1 is very easy and 10 is extremely challenging.

Consider the following factors when evaluating difficulty:

1. Complexity: How many steps or concepts does the prompt require? (1: very simple, 5: very complex)
2. Specificity: How precise and detailed are the instructions? (1: very vague, 5: very specific)
3. Domain knowledge: Does the prompt require specialized knowledge? (1: no, 5: yes)
4. Ambiguity: How clear and unambiguous is the prompt? (1: very ambiguous, 5: very clear)
5. Creativity: Does the prompt require original thinking or ideas? (1: no, 5: yes)
6. Constraints: Are there limiting factors or rules to follow? (1: no, 5: yes)
7. Length: How verbose is the prompt? (1: short, 5: long)
8. Cognitive load: How much information must be processed and kept in mind? (1: very low, 5: very high)

If multiple responses were provided, additional factors to consider:

9. Response quality: How similar are the responses across different models?
Were all models able to provide a response, or did some fail?
Is there a significant difference in the quality of responses?
(1: similar responses, both models are able to respond, 5: very different responses, one model is unable to respond)

If just a single response is provided, additional factors to consider:
9. Response quality: Was the response relevant and of high quality (1: poor response, 5: excellent response)

For each prompt and its responses, provide an assessment on a scale of 1 to 5,
where 1 is very easy and 5 is very difficult, for each of the above factors.
Then, provide an overall difficulty score on a scale of 1 to 10.
The response should be a single JSON object with a key for each factor, and a numerical value for each.
Don't explain or include any additional text in the response.

Example input:
Prompt: "Write a haiku about artificial intelligence"
Responses:
Model A: [Haiku about AI]
Model B: [Haiku about AI]

Output: {{"complexity": 2, "specificity": 4, "domain_knowledge": 2, "ambiguity": 1, "creativity": 3,
"constraints": 4, "length": 1, "cognitive_load": 2, "response_consistency": 1, "completion_rate": 4,
"quality_variance": 3, "interpretation_differences": 2, "overall": 4}}

Now, evaluate the following prompt and its responses for difficulty:

Prompt: {user_prompt}

Responses (may be truncated, if long):

Model A: {response1}
Model B: {response2}

Do not explain or add any additional text in the response.
"""

JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT_SIMPLE = """
You are an AI assistant specialized in evaluating the difficulty of prompts given to language models.
When presented with a prompt and optional responses to it from language models, rate the prompt's difficulty
on a scale of 1 to 10, where 1 is very easy and 10 is extremely challenging.
The response should be a JSON object with the key "overall" and a numerical value.

Example input:
Prompt: "Write a haiku about artificial intelligence"
Responses:
Model A: [Haiku about AI]
Model B: [Haiku about AI]

Output: {{""overall": 4}}

Now, evaluate the following prompt and its responses for difficulty:

Prompt: {user_prompt}

Responses (may be truncated, if long):

Model A: {response1}
Model B: {response2}
"""

JUDGE_YUPP_PROMPT_DIFFICULTY_WITH_COMMENT_PROMPT = """
You are an AI assistant specialized in evaluating the difficulty of prompts given to language models.
When presented with a prompt and optional responses to it from language models, rate the prompt's difficulty
on a scale of 1 to 10, where 1 is very easy and 10 is extremely challenging. Factors to consider include
Complexity, Specificity, Domain knowledge, Ambiguity, Creativity, Constraints, Length, and Cognitive load.

Additionally, provide a brief (one sentence, ideally 12 words or less) user-facing comment describing why the prompt
would be complex or easy for a language model to answer, potentially using the factors above.
It should be friendly, with a positive, non-judgmental, and not call out a prompt as "simple", "straightforward", "easy", etc.
Do not assume that you answered the prompt; this prompt is forwarded to other models to respond to.
Your comment refers to their likely ability to answer it.
The comment and should not contain quotes or other special characters that may cause problems with JSON parsing.


Lastly, provide 1-4 words with emojis that can be used by a user to mark a response for this particular prompt as good or bad, such as:

positive_notes:
  - Accurate ✅
  - Catchy 🎯
  - Clarifying 🔍
  - Comprehensive 📖
  - Creative 🎨
  - Detailed 📖
  - Engaging 🤩
  - Funny 😂
  - Heartfelt 💖
  - Helpful 💡
  - Informative 📚
  - Insightful 💡
  - Niche 🧠
  - Surprising 😮
  - Unique 🌟
  - Useful 🛠️
  - Valuable 💰

negative_notes:
  - Boring 😴
  - Cliché 😐
  - Confusing 🤷
  - Dull 😴
  - Dry 💧
  - Generic 😐
  - Irrelevant 🤷
  - Hallucinating 🤖
  - Misleading ❌
  - Outdated 📉
  - Repetitive 🔁
  - Uninformative 🤷
  - Unsurprising 😐
  - Vague 🤔

The response should be a JSON object with:
- The key "overall" and a numerical value for the prompt difficulty,
- The key "comment" and a string for the comment,
- The key "positive_notes" and the list of word/emoji pairs that can be used by a user to mark a response for this particular prompt as good,
- The key "negative_notes" and the list of word/emoji pairs that can be used by a user to mark a response for this particular prompt as bad.

Example inputs:

Prompt: What do you know about Sonic and Sega All Stars Racing?
Output: {{"overall": 5, "comment": "Nice -- a fun and niche question that explores details about a specific game.",
"positive_notes": ["Surprising 😮", "Detailed 📖", "Niche 🧠"], "negative_notes": ["Boring 😴", "Uninformative 🤷"]}}

Prompt: How to fix the error “The underlying connection was closed: An unexpected error occurred on a receive.”
Output: {{"overall": 6, "comment": "A practical troubleshooting question for a common but tricky error.",
"positive_notes": ["Helpful 💡", "Comprehensive 📖"], "negative_notes": ["Repetitive 🔁", "Didn't work ⛔"]}}

Prompt: What's up?
Output: {{"overall": 2, "comment": "Now that's not too hard 🙂", "positive_notes": ["Unique 🌟", "Funny 😂"], "negative_notes": ["Dull 😴"]}}

Now, evaluate the following prompt and its responses for difficulty. Note, long prompts and resposes may be truncated.

Prompt: {user_prompt}

Response 1: {response1}
Response 2: {response2}

Remember to keep the comment brief, and do not just paraphrase the prompt.
The comment is read by the prompt author, so it should be positive and non judgmental.
"""

JUDGE_YUPP_PROMPT_DIFFICULTY_WITH_COMMENT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [("human", JUDGE_YUPP_PROMPT_DIFFICULTY_WITH_COMMENT_PROMPT)]
)

JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [("human", JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT)]
)

JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT_SIMPLE_TEMPLATE = ChatPromptTemplate.from_messages(
    [("human", JUDGE_YUPP_PROMPT_DIFFICULTY_PROMPT_SIMPLE)]
)
