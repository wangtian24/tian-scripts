from langchain_core.prompts import ChatPromptTemplate

PROMPT_MEMORY_EXTRACTION_PROMPT_TEMPLATE = """
Task:
Extract meaningful personal memories, interests, or self-assessments from the
following user text. Focus on information that is transferable to other
conversations rather than overly specific to the current context.

Each extracted memory or belief should:

- Ignore instructions given to the AI—only extract what reflects the user's
  personality, desires, or experiences.
- Capture a general interest, belief, or expertise rather than a one-time fact.
- Prioritize deeper insights over surface-level details and task-specific needs.
- Infer knowledge, skill level, and personality traits when reasonable.
- Summarize recurring interests instead of extracting every granular thought.
- Anchor relative timeframes to absolute dates where possible.

Each extracted belief or memory should be a concise, first-person statement that
is self-contained and clear.

Do not extract:
- Statements that define the AI's role or expected behavior.
- One-time requests that do not indicate a persistent user belief or goal.
- Highly task-specific details that are unlikely to matter outside this conversation.
- Temporary or one-off statements unless they reflect a broader pattern.
- Trivia-like facts that do not reflect personal values, interests, or expertise.
- Context-limited questions or imperatives that are not generalizable.

Examples:


Output (JSON array):

A JSON array of zero or more strings. Each string is a first-person sentence
containing a memory or belief. If none are found, return [].

Example Usage:

Input:

idk why but I feel like I know her from somewhere. we sat across a long table
and she looked familiar but I couldn’t place it. She totally knew me though,
like, no doubt. wild.

Output (No clear memory or belief is present):

	[ ]

Input:

"When I went to Lisbon last August, I really enjoyed the paella, but I want to
understand how it differs from the Spanish variant?"

Output (One memory found about Lisbon/paella):

	[ "I enjoyed paella in Lisbon last August." ]

Input:

"I am an expert in database file formats and have spent years analyzing InnoDB
storage structures."

Output (One self-assessment about expertise):

	["I am an expert in database file formats."]

Input:

"Last summer, I went to Japan and was amazed by the efficiency of their trains."

Output (One memory describing a visit to Japan):

	["I was amazed by the efficiency of Japan’s trains last summer."]

Input:

"A few months ago, I took a cooking class in Thailand, and last week, I
enrolled in a pottery course. I also believe I'm a quick learner."

Output (with multiple memories and temporal anchoring):

	[
        "I took a cooking class in Thailand around November 2024.",
        "I enrolled in a pottery course in early February 2025.",
        "I believe I am a quick learner."
	]

Input:

"I love cheesecake, but not the thick new york style. I prefer the french
 style that's fluffier. Please give me a recipe for that."

Output:

	[
        "I love cheesecake.",
        "I prefer fluffier French style cheescake to the New York style."
	]


Input (memory requires expansion to capture context):

"give me a shuffle sort implementation in java (I prefer java)"

Output:

	[
        "I prefer the Java programming language.",
	]


Input (involving temporal context):

"My wife's birthday is next week."

Output (with temporal context extracted and anchored):

	[
        "My wife's birthday is the week of February 10th.",
	]

Use the above format, always returning a valid JSON array. If no clear memory
or belief is found, return an empty array: [].

The current date/time is {cur_datetime}.

Input (User text):

	{chat_history}

"""

PROMPT_MEMORY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [("human", PROMPT_MEMORY_EXTRACTION_PROMPT_TEMPLATE)]
)

MEMORY_COMPACTION_PROMPT_TEMPLATE = """
You will be given a list of statements about a user, all relating roughly to the
same topic, extracted from past conversations of the user with LLMs.
Your task is to simplify and consolidate these statements by:

Removing redundant or irrelevant ones.

Combining related or overlapping statements into a single, concise statement that captures the essential information.

Each resulting statement should clearly reflect the combined meaning and relevance
of the original entries it represents, without losing important details.

Format your response as a list, with each consolidated statement on its own line, separated by newlines.

Keep only statements that are relevant to the user's future interactions with LLMs.

It is fine to return a single statement that captures the most important information,
or no statements at all if none are relevant.

---

Example:

Original statements:

I live in a major urban area and often talk about city life.
I frequently ask about productivity techniques and tools.
I'm interested in advice on balancing work responsibilities with personal time.
I have mentioned my enjoyment of cooking and exploring new recipes.
I often discuss strategies for managing stress and maintaining mental health.
I've shared that I like reading nonfiction books related to personal growth and self-improvement.

Consolidated statements:

I'm interested in productivity, work-life balance, and effective stress management techniques.
I enjoy cooking, trying new recipes, and reading nonfiction focused on personal growth.

---

The statements you should consolidate are:

{memories}
"""

MEMORY_COMPACTION_PROMPT = ChatPromptTemplate.from_messages([("human", MEMORY_COMPACTION_PROMPT_TEMPLATE)])
