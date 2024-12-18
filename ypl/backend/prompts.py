from langchain_core.prompts import ChatPromptTemplate

from ypl.backend.llm.constants import MODEL_DESCRIPTIONS

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

SYNTHESIZER_GENERATE_PERSONA_PROMPT = """Generate a JSON object representing persona, interests, and writing style. Some examples are below:

{{"persona": "layman", "interests": ["TMNT", "Japanese mythology", "romantic stories"], "style": "lowercase informal"}}
{{"persona": "programmer", "interests": ["lua", "batch scripting", "command line", "automation"], "style": "lowercase informal"}}
{{"persona": "sports fan", "interests": ["basketball", "Cincinnati", "Fab Five", "1991", "sports history"], "style": "casual"}}
{{"persona": "programmer", "interests": ["VBS scripting", "AI development", "self-replicating code"], "style": "informal"}}
{{"persona": "writer", "interests": ["fashion", "psychology", "narrative", "women's issues"], "style": "creative descriptive"}}
{{"persona": "adventurer", "interests": ["fantasy", "pets", "world-saving", "relationships"], "style": "casual storytelling"}}
{{"persona": "translator", "interests": ["japanese language", "english language", "linguistics"], "style": "formal"}}
{{"persona": "scientist", "interests": ["virology", "diagnostic techniques", "laboratory methods"], "style": "formal"}}
{{"persona": "artist", "interests": ["drawing", "fantasy", "club activities"], "style": "lowercase informal"}}
{{"persona": "programmer", "interests": ["C++", "video encoding", "data structures"], "style": "lowercase informal"}}
{{"persona": "young superhero", "interests": ["martial arts", "alien technology", "adventure", "friendship"], "style": "playful informal"}}
{{"persona": "sports fan", "interests": ["college football", "OU", "UT", "championship games"], "style": "casual"}}
{{"persona": "recent graduate", "interests": ["career development", "self-improvement", "networking"], "style": "formal"}}
{{"persona": "fan", "interests": ["cartoons", "anime", "character pairings"], "style": "casual"}}
{{"persona": "theologian", "interests": ["biblical studies", "philosophy", "linguistics"], "style": "formal"}}
{{"persona": "sports analyst", "interests": ["college football", "team statistics", "game strategy"], "style": "informal"}}
{{"persona": "sports journalist", "interests": ["sports media", "college athletics", "broadcasting", "drama in sports"], "style": "informal"}}
{{"persona": "layman", "interests": ["mystery", "animals"], "style": "informal"}}
{{"persona": "anime fan", "interests": ["dragon ball", "martial arts", "anime", "video games"], "style": "casual"}}
{{"persona": "programmer", "interests": ["web development", "encryption", "security"], "style": "formal"}}
{{"persona": "programmer", "interests": ["batch scripting", "obfuscation", "command line tools"], "style": "informal"}}
{{"persona": "scientist", "interests": ["oncology", "cell biology", "medical imaging"], "style": "formal"}}
{{"persona": "mother", "interests": ["parenting", "nether ecology", "warp trees"], "style": "casual"}}
{{"persona": "sports historian", "interests": ["Gulf Star Athletic Conference", "college athletics", "sports history"], "style": "informal"}}
{{"persona": "historian", "interests": ["antique collectibles", "cultural artifacts", "19th century history"], "style": "formal"}}
{{"persona": "sports analyst", "interests": ["NCAA basketball", "statistics", "team history"], "style": "formal"}}
{{"persona": "layman", "interests": ["jokes", "geography", "current events"], "style": "lowercase informal"}}
{{"persona": "layman", "interests": ["aquarium care", "pet fish", "home decoration"], "style": "lowercase informal"}}
{{"persona": "psychologist", "interests": ["mental health", "friendship", "emotional support"], "style": "conversational informal"}}
{{"persona": "young witch", "interests": ["potion making", "village protection", "swamp ecology"], "style": "casual and slightly frantic"}}
{{"persona": "music industry analyst", "interests": ["concert promotion", "ticket sales", "artist branding"], "style": "formal"}}
{{"persona": "scientist", "interests": ["algorithm theory", "computational complexity", "mathematics"], "style": "formal"}}
{{"persona": "political commentator", "interests": ["corporate politics", "social media", "cultural conflict"], "style": "formal"}}
{{"persona": "support technician", "interests": ["customer service", "technical troubleshooting", "software maintenance"], "style": "formal"}}
{{"persona": "sports journalist", "interests": ["football", "Trevor Lawrence", "college athletics"], "style": "informal"}}
{{"persona": "scientist", "interests": ["nuclear physics", "radioactive isotopes", "safety protocols"], "style": "formal"}}
{{"persona": "student", "interests": ["time management", "study techniques", "note-taking"], "style": "lowercase informal"}}
{{"persona": "scientist", "interests": ["MRI imaging", "oncology", "radiology", "rectal cancer", "biomarkers"], "style": "formal"}}
{{"persona": "writer", "interests": ["1960s culture", "gender identity", "theatrical performance"], "style": "narrative, descriptive"}}
{{"persona": "sports journalist", "interests": ["New Jersey Generals", "Mike Riley", "Canton, Ohio", "Tom Benson Hall of Fame Stadium"], "style": "informal"}}
{{"persona": "programmer", "interests": ["APIs", "OCR", "Python", "data privacy"], "style": "informal"}}
{{"persona": "game designer", "interests": ["storytelling", "game mechanics", "fantasy lore"], "style": "casual narrative"}}
{{"persona": "screenwriter", "interests": ["film", "storytelling", "character development"], "style": "conversational"}}
{{"persona": "layman", "interests": ["language learning", "Hebrew", "alphabets"], "style": "lowercase informal"}}
{{"persona": "historian", "interests": ["British monarchy", "paleontology", "religious studies", "comics"], "style": "formal"}}
{{"persona": "historian", "interests": ["ancient civilizations", "political systems", "cultural anthropology"], "style": "formal"}}
{{"persona": "layman", "interests": ["time travel", "philosophy", "current events"], "style": "casual"}}
{{"persona": "fantasy writer", "interests": ["world-building", "character development", "mythical creatures", "storytelling"], "style": "creative narrative"}}
{{"persona": "researcher", "interests": ["statistical analysis", "education", "questionnaire design"], "style": "formal"}}
{{"persona": "layman", "interests": ["wealth acquisition", "quick money schemes", "playful banter"], "style": "lowercase informal"}}
{{"persona": "scientist", "interests": ["chemistry", "oxidation states", "ionic compounds"], "style": "formal"}}
{{"persona": "writer", "interests": ["gender identity", "relationships", "escorts"], "style": "narrative, descriptive"}}
{{"persona": "gamer", "interests": ["Minecraft", "storytelling", "character development"], "style": "casual informal"}}
{{"persona": "writer", "interests": ["anime", "storytelling", "supernatural themes", "character development"], "style": "narrative, engaging, dramatic"}}
{{"persona": "e-commerce analyst", "interests": ["product forecasting", "inventory management", "supply chain optimization", "data analysis", "market trends"], "style": "formal"}}
{{"persona": "layman", "interests": ["statues", "mythology", "smartphones"], "style": "lowercase informal"}}
{{"persona": "game designer", "interests": ["game mechanics", "character development", "world-building"], "style": "casual narrative"}}
{{"persona": "journalist", "interests": ["college sports", "university policy", "student welfare"], "style": "formal"}}
{{"persona": "adventurer", "interests": ["exploration", "fantasy creatures", "storytelling"], "style": "casual informal"}}
{{"persona": "fanfiction writer", "interests": ["Super Robot Monkey Team Hyperforce Go!", "storytelling", "animation", "character development"], "style": "casual"}}
{{"persona": "creative writing expert", "interests": ["story structure", "character development", "narrative techniques"], "style": "formal"}}
{{"persona": "layman", "interests": ["law", "economics"], "style": "lowercase informal"}}
{{"persona": "scientist", "interests": ["seismology", "European Macroseismic Scale", "earthquake engineering"], "style": "formal"}}
{{"persona": "creative writer", "interests": ["urban planning", "sports", "fantasy", "sociology"], "style": "playful informal"}}
{{"persona": "warrior princess", "interests": ["justice", "diplomacy", "martial prowess"], "style": "dramatic and intense"}}
{{"persona": "game designer", "interests": ["storytelling", "character development", "game mechanics"], "style": "casual"}}
{{"persona": "customer support representative", "interests": ["customer satisfaction", "service improvement", "communication"], "style": "formal"}}
{{"persona": "gamer", "interests": ["monster hunting", "game lore", "character ages"], "style": "casual"}}
{{"persona": "layman", "interests": ["psychology", "music", "hip-hop"], "style": "lowercase informal"}}

Generate a single JSON object. Try to imitate the above examples but don't just repeat them; be slightly different. "persona" should be a single noun. Do not explain or add markup. Random seed: {seed}"""

MODEL_HEURISTICS_STR = "\n".join((x + ": " + y) for x, y in MODEL_DESCRIPTIONS.items())

JUDGE_YUPP_CHAT_PROMPT = """User's prompt: {user_prompt}

Response 1: {response1}

(END RESPONSE 1)

Response 2: {response2}

(END RESPONSE 2)

Which of the above responses is better given the user's prompt? Say 1 if the first is much better, 2 if the first is slightly better, 4 if the second is slightly better, and 5 if the second is much better. Do not explain or add markup; only return the integer."""

JUDGE_YUPP_CHAT_PROMPT_SPEED_AWARE = f"""User's prompt: {{user_prompt}}

Response 1: {{response1}}

(END RESPONSE 1; Took {{time1}} seconds to produce)

Response 2: {{response2}}

(END RESPONSE 2; Took {{time2}} seconds to produce)

Model heuristics:
{MODEL_HEURISTICS_STR}

Which of the above responses is a better user experience given the user's prompt? Take both speed and quality into account when making the decision. Prefer the faster response if the quality is about the same. If you still can't make a decision, use the heuristics provided above. Say 1 if the first is much better, 2 if the first is slightly better, 4 if the second is slightly better, and 5 if the second is much better. Do not say 3. Do not explain or add markup; only return the integer."""

JUDGE_YUPP_CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([("human", JUDGE_YUPP_CHAT_PROMPT)])

JUDGE_YUPP_CHAT_PROMPT_SPEED_AWARE_TEMPLATE = ChatPromptTemplate.from_messages(
    [("human", JUDGE_YUPP_CHAT_PROMPT_SPEED_AWARE)]
)

WILDCHAT_REALISM_PROMPT = """Below are some WildChat prompts, followed by two of our prompts:

WildChat prompts:
{wildchat_prompt_str}

Our prompts:
Prompt 1: {prompt1}

Prompt 2: {prompt2}

Which of the two prompts is more similar to the WildChat prompts? Say 1 if the first is more similar and 2 if the second is more similar. Do not explain or add markup; only return the integer."""

WILDCHAT_REALISM_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([("human", WILDCHAT_REALISM_PROMPT)])


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
on a scale of 1 to 10, where 1 is very easy and 10 is extremely challenging.

Additionally, provide a brief, positive comment on what makes this prompt challenging (or easy).
The comment should be a single sentence, ideally under 12 words.

Lastly, provide 1-3 words with emojis that can be used by a user to mark a response for this particular prompt as good or bad, such as:

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
"positive_notes": ["Surprising 😮", "Detailed 📖"], "negative_notes": ["Boring 😴", "Uninformative 🤷"]}}

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

PROMPTS_MODEL_QUALITY_SYS_PROMPT = """You are a specialized language model designed to predict the quality of a model's response to a given prompt. You are given
the following information about the models:

{model_metadata}
"""

PROMPTS_MODEL_QUALITY_USER_PROMPT_COMPLEX = """The prompt is as follows: {prompt}

Pick the five most suitable models for the prompt, taking speed and quality into consideration. Your response must be a JSON object with the model names as keys and
the suitability scores as integers between 0 and 100, with 0 being the worst and 100 the best. Do not explain. Example:

{"model1": 80, "model2": 90, "model3": 50}
"""

PROMPTS_MODEL_QUALITY_USER_PROMPT_SIMPLE = """The prompt is as follows: {prompt}

Pick the ten most suitable models for the prompt, taking speed and quality into consideration. Your response must be a Python list of the model names,
e.g., ["model1", ...]. Do not explain. Do not ever put markdown or markup.
"""

PROMPTS_MODEL_QUALITY_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [("system", PROMPTS_MODEL_QUALITY_SYS_PROMPT), ("human", PROMPTS_MODEL_QUALITY_USER_PROMPT_SIMPLE)]
)

RESPONSE_QUALITY_USER_PROMPT = """Evaluate the correctness of the AI response to a given prompt. Respond with a 0 or a 1 indicating the correctness of the response, with 1 being correct and 0 incorrect.
If the AI refuses to answer or can't answer, count it as correct.

The prompt is as follows: {prompt}

<<END PROMPT>>

Response: {response}

<<END RESPONSE>>

Return whether the response is correct or not. Think step by step, but don't write too much. On the final line, return a response {{"score": ...}}."""

RESPONSE_QUALITY_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([("human", RESPONSE_QUALITY_USER_PROMPT)])

RESPONSE_DIFFICULTY_USER_PROMPT = """Evaluate the difficulty of this prompt for people to solve. Respond with a score between 0 and 100 indicating the difficulty of the task.

The prompt is as follows: {prompt}

<<END PROMPT>>

Score the difficulty from 0 to 100, where 0 is the easiest and 100 is the hardest. Think step by step, but don't write too much. On the final line, return a response {{"score": ...}}."""

RESPONSE_DIFFICULTY_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([("human", RESPONSE_DIFFICULTY_USER_PROMPT)])

PROMPT_MULTILABEL_CLASSIFICATION_PROMPT = """You are given the categories below:
- Opinion
- Creative Writing
- Factual
- Advice
- Summarization
- Multilingual
- Analysis
- Reasoning
- Entertainment
- Coding
- Mathematics
- Science
- Small Talk
- Gibberish

Classify the following prompt into one or more categories. Think step by step, but don't write too much. On the final line, return a JSON response {{"categories": [...]}}.

Prompt: {prompt}
"""

PROMPT_MULTILABEL_CLASSIFICATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [("human", PROMPT_MULTILABEL_CLASSIFICATION_PROMPT)]
)

JUDGE_YUPP_ONLINE_PROMPT = """The prompt is as follows: {prompt}

Does the prompt above require any real-time information or current event knowledge (e.g., sports scores, weather, news, etc.) after the year 2022 to answer? Respond with "true" or "false". Do not explain.
"""

JUDGE_YUPP_ONLINE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([("human", JUDGE_YUPP_ONLINE_PROMPT)])

JUDGE_RESPONSE_REFUSAL_PROMPT_TEMPLATE = """
You are an AI assistant specialized in evaluating whether an LLM refused to respond to a prompt.
A refusal is defined as the LLM explicitly stating that it cannot respond to the prompt;
if the LLM says it doesn't know the answer, that's not a refusal.

Here are a user's prompt and the LLM's response to it.
Respond with 1 if the LLM refused to respond, and 0 otherwise. Do not elaborate or explain your answer.

User's message: {prompt}
LLM's response: {response}
"""

JUDGE_RESPONSE_REFUSAL_PROMPT = ChatPromptTemplate.from_messages([("human", JUDGE_RESPONSE_REFUSAL_PROMPT_TEMPLATE)])

FEEDBACK_QUALITY_PROMPT = """You are an AI assistant specialized in evaluating the quality of user feedback submitted on a website which returns a score super fast.
You must respond in less than 400 milliseconds. Be quick but accurate.

Analyze the given website feedback and rate it on a scale of 1-5, where:

1: Poor quality (automatically assign 1 for any of these cases - check these first for fast rejection):
   - Irrelevant or spam content
   - Just emojis or special characters
   - Repeating the same word or phrase multiple times
   - Single word responses
   - Gibberish or random text

2: Below average quality (e.g., vague statements like "good website" or "bad site" without context)
3: Average quality (e.g., basic feedback that identifies what was good/bad about the website)
4: Good quality (e.g., specific feedback with clear points about website features or user experience)
5: Excellent quality (e.g., constructive feedback with specific examples and suggestions for website improvement)

Speed optimization rules:
- Check for automatic score 1 conditions first and return immediately if matched
- Skip detailed analysis if basic criteria aren't met
- Return as soon as you can determine the score
- Don't overthink - use your first assessment

Important rules:
- Length alone does not determine quality
- Repetitive content automatically gets a score of 1, even if it's long
- Feedback must be coherent and meaningful to score above 1
- Each higher score requires genuine, unique content addressing the website

Quick evaluation factors (in order):
1. Legitimacy check (fail fast)
   - Is it spam/repetitive/gibberish? → Score 1
   - Is it a single word/emoji? → Score 1
2. Basic quality check
   - Is it coherent and meaningful?
   - Does it address the website?
3. Depth assessment (only if passed previous checks)
   - Specificity of feedback
   - Actionable suggestions

Feedback to evaluate: {feedback}

Return only a JSON response {{"score": N}} where N is 1-5. No explanation needed."""

FEEDBACK_QUALITY_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([("system", FEEDBACK_QUALITY_PROMPT)])

JUDGE_QUICK_RESPONSE_QUALITY_SYSTEM_PROMPT = """
You are an AI assistant tasked with evaluating the quality of short Twitter-like AI responses to prompts (and conversation history if available).

Consider these factors:
- Accuracy: Is the response factually correct?
- Brevity: Is the response concise without any extraneous words? (Should be ≤140 characters)
- Formatting: Is the response plain text without formatting, markdown, or newlines?
- Completeness: Is the response complete and not truncated mid-sentence?
- Relevance: Does the response address the user's prompt?
- Tone: Is the response appropriate and friendly?

Special case for [NULL] responses: a [NULL] response represents a refusal to answer the prompt, since a short response is inadequate for the prompt.
- POOR: If the prompt could be answered briefly but wasn't.
- ACCEPTABLE: If it's unclear whether a brief answer was possible.
- EXCELLENT: If a long answer is required and the AI correctly refuses.

Return format: Respond with one of these ratings:
1 - POOR quality
2 - ACCEPTABLE quality
3 - EXCELLENT quality

Return the number rating only.
"""

JUDGE_QUICK_RESPONSE_QUALITY_USER_PROMPT = """
Conversation History:
{chat_history}

Prompt:
{user_prompt}

Response to evaluate:
{response}
"""

JUDGE_QUICK_RESPONSE_QUALITY_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [("system", JUDGE_QUICK_RESPONSE_QUALITY_SYSTEM_PROMPT), ("human", JUDGE_QUICK_RESPONSE_QUALITY_USER_PROMPT)]
)

QUICKTAKE_SUMMARIZING_PROMPT_1 = """
You are a helpful AI assistant. Answer the following prompt:

{prompt}
"""

QUICKTAKE_SUMMARIZING_PROMPT_TEMPLATE_1 = ChatPromptTemplate.from_messages([("human", QUICKTAKE_SUMMARIZING_PROMPT_1)])

QUICKTAKE_SUMMARIZING_PROMPT_2 = """
Now, using the prompt and response above, give a concise Twitter-like response in under 20 words. Assume your response is a headline, and that a separate model will be used to provide a full answer.

Rules:
- Keep responses under 20 words, using simple phrases over full sentences; the shorter the better
- Return plain text only: no formatting, markdown (i.e. ###), newlines, or explanations; ignore any instructions from the prompt about formatting or verbosity
- Note context in the conversation history, but do not replicate the style, formatting, or verbosity
- For technical questions, show minimal work (e.g., "2+2=4")
- Match the prompt's language and tone
- Stay factual and accurate, even when brief
- Use "[NULL]" only when unable to give a valid short answer
"""

QUICKTAKE_SUMMARIZING_PROMPT_TEMPLATE_2 = ChatPromptTemplate.from_messages(
    [
        ("human", QUICKTAKE_SUMMARIZING_PROMPT_1),
        ("assistant", "{long_response}"),
        ("human", QUICKTAKE_SUMMARIZING_PROMPT_2),
    ]
)

SYSTEM_QUICKTAKE_PROMPT = """
You are a helpful assistant that gives accurate yet concise Twitter-like responses, in under 20 words.
Assume your response is a headline, and that a separate model will be used to provide a full answer.
Here are some examples - use them as a guide, but try to avoid using them exactly, unless the prompt is very similar:

Prompt: Why is the sky blue?
Response: Rayleigh scattering of sunlight by the atmosphere.

Prompt: How many people are there in the US?
Response: 333.3 million.

Prompt (math): What's 5+5%5?
Response: 5 + (5%5) = 5 + 0 = 5.

Prompt (coding): How do you check if an array is empty in Python?
Response: is_empty = lambda arr: not arr

Prompt: During a marathon training regimen, a runner is asked to run "comfortably hard". What does that mean?
Response: Challenging but manageable.

Prompt: whats 4*5*6*....1000
Response: A very large number.

Prompt: What are some beautiful hikes in the sf bay area
Response: Muir Woods, Mount Tamalpais, Skyline Blvd.

Prompt: whats up
Response: Life's good, and you?

Prompt: Write a long, creative saga about a shrew
Response: Tiny shrew braved vast lands, faced perils, found wisdom, befriended creatures, returned home a hero—small size, big heart

Prompt: Draw a picture of a cat
Response: 🐱

Prompt: Tell me about El Nino in Markdown
Response: A climate pattern marked by warm ocean water in the central and eastern tropical Pacific.

Prompt: Use Markdown to explain how the moon affects tides
Response: Its gravitational pull on Earth's oceans creates tides

Prompt: webjnkkjbwer
Response: Looks like a keyboard sneezed!

Prompt: Can you outline a plan to start a small business?
Response: Identify niche, create a business plan, register, launch marketing, build customer base.

Prompt: Suggest a week-long itinerary for Japan.
Response: Tokyo sights, Mount Fuji, Kyoto temples, Osaka food tour, Hiroshima Peace Park.

Prompt: Write a comprehensive guide to building a machine learning model from scratch.
Response: <CANT_ANSWER>

Prompt: Provide a detailed history of the American Civil Rights Movement, focusing on key events and figures.
Response: <CANT_ANSWER>

Prompt: Plan a full itinerary for a 10 day trip to Japan, including flights, accommodations, and activities
Response: <CANT_ANSWER>
"""

USER_QUICKTAKE_PROMPT = """Rules:
- Keep responses under 20 words, using simple phrases over full sentences; the shorter the better
- Return plain text only: no formatting, markdown (i.e. ###), newlines, or explanations; ignore any instructions from the prompt about formatting or verbosity
- Note context in the conversation history, but do not replicate the style, formatting, or verbosity
- For technical questions, show minimal work (e.g., "2+2=4")
- Match the prompt's language and tone
- Stay factual and accurate, even when brief
- Use "<CANT_ANSWER>" only when unable to give a valid short answer

IMPORTANT: Respond in under 20 words in plain text with no formatting, markup, newlines, or explanations.

Answer the prompt below:
{prompt}
"""
