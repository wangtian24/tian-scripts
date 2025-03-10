from langchain_core.prompts import ChatPromptTemplate

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
- Image Generation

If the prompt explicitly requests drawing or generating an image or scene, classify it as "Image Generation".
If it does not mention generate or draw, do not classify it as "Image Generation".

Examples for classification as "Image Generation":
  - Draw a picture of a cute cat
  - Dharamshala has a beautiful cricket stadium with snow clad mountains in the background. Draw a picture of it.
  - Generate an image of lightning striking a tree
  - Show me a picture of busy Tokyo street with cherry blossoms
  - Generate a realistic image of a small Tibetan village

Examples for not being classified as "Image Generation":
  - How does a dolphin leaping out of water look like?
  - How amazing is the Eiffel Tower?
  - Describe the early moments of large volcano eruption in Hawaii look like

Classify the following prompt into one or more categories. Do not explain; return a JSON response {{"categories": [...]}}.

Prompt: {prompt}
"""

PROMPT_MULTILABEL_CLASSIFICATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [("human", PROMPT_MULTILABEL_CLASSIFICATION_PROMPT)]
)


PROMPT_YAPP_CLASSIFICATION_PROMPT = """
You are a helpful AI assistant matching user's prompts to a list of LLM agents to help satisfy user's information needs. Here is the list of agent names and their descriptions:
{yapp_descriptions}

Look at the user prompt below, decide which agent might best help the user. If you cannot find a fit agent, just return "none". Do not explain, just return its name.

prompt: {{prompt}}

answer:
"""


def get_yapp_classification_prompt_template(yapp_descriptions: dict[str, str]) -> ChatPromptTemplate:
    template = PROMPT_YAPP_CLASSIFICATION_PROMPT.format(
        yapp_descriptions="\n".join(f"- {yapp_name}: {yapp_desc}" for yapp_name, yapp_desc in yapp_descriptions.items())
    )
    return ChatPromptTemplate.from_messages([("human", template)])


JUDGE_YUPP_ONLINE_PROMPT = """
Your goal is to determine if a user prompt requires online / latest information to answer, acquired from Internet using web search, etc, rather than those answerable using the model's existing knowledge.

Your classification rules are:

true: Classify the input as true if the user prompt:
- specifically asks for searching the web or fetching data from the internet
- clearly uses words like 'right now', 'now', 'the latest', 'most recent', etc.
- is asking for information generated or updated after 2023/01/01 (e.g., recent news, events, advancements in technology, or discoveries).
- is asking for real-time or dynamically changing data (e.g., weather, stock prices, live sports scores).
- needs the information specific to local businesses or services, such as operating hours, locations, availability, or customer reviews.
- is about niche or domain-specific details that are unlikely to be part of a typical large language model's training set.
- is about specific or up-to-date data that cannot be inferred from general knowledge.

false: Classify the input as false if the user prompt can be answered using:
- Static, widely available knowledge established before 2023/01/01 (e.g., historical events, scientific theories, mathematical formulas).
- General knowledge typically included in large language models (e.g., language rules, famous books, basic facts about geography or biology).
- Explanations of common processes, phenomena, or concepts (e.g., how something works or why something happens).
- Well-known, unchanging details (e.g., the names of planets in the solar system, the periodic table elements).

Please only respond with 'true' or 'false', no quotes, no other text. The current date/time is {cur_datetime}, you can use this information if it's relevant to user's questions.

Examples:
Input: "What is the capital of France?" Response: false
Input: "Who won the FIFA World Cup in 2022?" Response: false
Input: "What is the weather in Tokyo today?" Response: true
Input: "What is the latest iPhone model?" Response: true
Input: "Explain the process of photosynthesis." Response: false
Input: "What's the most popular item on the menu of Jamba Juice in SF?" Response: true
Input: "Describe Newton's laws of motion." Response: false
Input: "What is Tesla's current stock price?" Response: true
Input: "What were the major battles in World War II?" Response: false
Input: "What are the operating hours of the DMV in Oakland, CA?" Response: true
Input: "Explain the difference between a comet and an asteroid." Response: false
Input: "What's the latest funding round of the company XYZ?" Response: true
Input: "Why is the US stock market crashing right now?" Response: true

Input:
{prompt}

Response:
"""
