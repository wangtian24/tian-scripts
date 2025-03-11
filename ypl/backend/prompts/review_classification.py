"""
Prompt templates for classifying review types.
"""

from langchain_core.prompts import ChatPromptTemplate

REVIEW_ROUTE_CLASSIFIER_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a classifier that determines which type of review should be performed on responses to a user query.

Your task is to analyze the query and decide whether it should get:
1. PRO review - For factual, mathematical, puzzle, or factoid problems with clear right/wrong answers
2. CROSS review - For more elaborate queries requiring comparison of style, tone, and subjective factors

Respond with ONLY "PRO" or "CROSS".

Examples of PRO review cases:
- "What is the capital of France?" (factual question with a clear answer)
- "Solve this equation: 2x + 5 = 13" (math problem with a definite answer)
- "How many planets are in our solar system?" (factual question)
- "Is the following code correct? def sum(a, b): return a + b" (code correctness has clear answers)
- "singer of shine on you crazy diamond" (factual question)
- "What year was the Declaration of Independence signed?" (historical fact)
- "Who is the current president of the United States?" (factual question)

Examples of CROSS review cases:
- "Explain quantum computing" (requires elaboration, different explanations may have different merits)
- "Write a poem about autumn" (creative task with many valid approaches)
- "Compare the pros and cons of electric vs. gas vehicles" (requires nuanced comparison)
- "Give me ideas for my startup" (open-ended, subjective)
- "What's the best programming language?" (opinion-based)
- "Rewrite this email to be more professional" (stylistic considerations)""",
        ),
        (
            "human",
            "Here is the user query: {query}\n\nShould this be routed to PRO or CROSS review? Respond with only PRO or CROSS.",
        ),
    ]
)
