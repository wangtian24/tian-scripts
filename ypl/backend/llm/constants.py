import re
from collections import defaultdict
from enum import Enum

from ypl.backend.llm.model_heuristics import ModelHeuristics


class ChatProvider(Enum):
    OPENAI = 1
    ANTHROPIC = 2
    GOOGLE = 3
    MISTRAL = 4
    META = 5
    MICROSOFT = 6
    ZERO_ONE = 7
    DEEPSEEK = 8
    NVIDIA = 9
    QWEN = 10
    HERMES = 11
    TOGETHER = 12
    ANYSCALE = 13
    HUGGINGFACE = 14
    AI21 = 15

    @classmethod
    def from_string(cls, provider: str) -> "ChatProvider":
        try:
            return cls[provider.upper()]
        except KeyError as e:
            raise ValueError(f"Unsupported provider string: {provider}") from e


# These are the models that are currently active in the system.
# TODO: This should be fetched from the database.
ACTIVE_MODELS_BY_PROVIDER = {
    ChatProvider.OPENAI: [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "o1-preview-2024-09-12",
        "o1-mini-2024-09-12",
    ],
    ChatProvider.MISTRAL: ["mistral-large-latest"],
    ChatProvider.GOOGLE: ["gemini-1.5-pro"],
    ChatProvider.ANTHROPIC: ["claude-3-5-sonnet-20240620"],
    ChatProvider.MICROSOFT: [
        "phi-3-mini-4k-instruct",
    ],
    ChatProvider.DEEPSEEK: [
        "deepseek-coder-v2",
    ],
    ChatProvider.QWEN: ["qwen-max"],
    ChatProvider.ANYSCALE: [
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ],
    ChatProvider.TOGETHER: [
        "gemma-2-9b-it",
        "qwen1.5-72b-chat",
    ],
}

ALL_MODELS_BY_PROVIDER = {
    ChatProvider.OPENAI: [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4o-mini-2024-07-18",
        "o1-preview-2024-09-12",
        "o1-mini-2024-09-12",
    ],
    ChatProvider.MISTRAL: ["mistral-large-latest"],
    ChatProvider.GOOGLE: ["gemini-1.5-pro"],
    ChatProvider.ANTHROPIC: ["claude-3-5-sonnet-20240620"],
    ChatProvider.MICROSOFT: [
        "phi-3-mini-4k-instruct",
    ],
    ChatProvider.DEEPSEEK: [
        "deepseek-coder-v2",
    ],
    ChatProvider.QWEN: ["qwen-max"],
    ChatProvider.ANYSCALE: [
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ],
    ChatProvider.TOGETHER: [
        "gemma-2-9b-it",
        "qwen1.5-72b-chat",
    ],
}

ALL_EMBEDDING_MODELS_BY_PROVIDER = {ChatProvider.OPENAI: ["text-embedding-ada-002"]}

# fmt: off
FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hannah", "Isaac", "Julia",
    "Karl", "Linda", "Michael", "Nancy", "Oliver", "Pamela", "Quinn", "Rachel", "Steve", "Tina",
    "Ulysses", "Victoria", "Walter", "Xena", "Yvonne", "Zach", "Ava", "Ben", "Cathy", "Dylan"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
    "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez",
    "Robinson", "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "Hernandez",
    "King", "Wright", "Lopez", "Hill", "Scott", "Green", "Adams", "Baker", "Gonzalez", "Nelson",
    "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell", "Parker", "Evans", "Edwards"
]
# fmt: on
MODEL_HEURISTICS: defaultdict[str, ModelHeuristics] = defaultdict(
    lambda: ModelHeuristics(
        dollars_per_million_input_tokens=1,
        dollars_per_million_output_tokens=1,
        tokenizer_name="gpt-4o",
        tokens_per_second=75,
        skills={
            "all": 11,
            "other": 11,
            "opinion": 11,
            "advice": 11,
            "creative writing": 8,
            "mathematics": 5,
            "coding": 5,
            "analysis": 6,
            "entertainment": 11,
            "reasoning": 7,
            "multilingual": 5,
            "summarization": 6,
            "factual": 6,
            "science": 8,
            "small talk": 16,
            "gibberish": 18,
        },
    )
)

MODEL_HEURISTICS.update(
    {
        "gpt-4o": ModelHeuristics(
            dollars_per_million_input_tokens=5,
            dollars_per_million_output_tokens=15,
            tokenizer_name="gpt-4o",
            tokens_per_second=120,
            skills={
                "all": 15,
                "other": 11,
                "opinion": 12,
                "advice": 14,
                "creative writing": 12,
                "mathematics": 15,
                "coding": 15,
                "analysis": 16,
                "entertainment": 16,
                "reasoning": 11,
                "multilingual": 8,
                "summarization": 13,
                "factual": 16,
                "science": 16,
                "small talk": 13,
                "gibberish": 20,
            },
        ),
        "gpt-4o-mini": ModelHeuristics(
            dollars_per_million_input_tokens=0.15,
            dollars_per_million_output_tokens=0.6,
            tokenizer_name="gpt-4o-mini",
            tokens_per_second=100,
            skills={
                "all": 14,
                "other": 8,
                "opinion": 10,
                "advice": 10,
                "creative writing": 10,
                "mathematics": 12,
                "coding": 15,
                "analysis": 10,
                "entertainment": 11,
                "reasoning": 12,
                "multilingual": 6,
                "summarization": 20,
                "factual": 10,
                "science": 10,
                "small talk": 16,
                "gibberish": 10,
            },
        ),
        "gpt-4-turbo": ModelHeuristics(
            dollars_per_million_input_tokens=10,
            dollars_per_million_output_tokens=30,
            tokenizer_name="gpt-4-turbo",
            tokens_per_second=50,
            skills={
                "all": 15,
                "other": 10,
                "opinion": 14,
                "advice": 14,
                "creative writing": 12,
                "mathematics": 15,
                "coding": 15,
                "analysis": 16,
                "entertainment": 12,
                "reasoning": 13,
                "multilingual": 8,
                "summarization": 16,
                "factual": 16,
                "science": 16,
                "small talk": 18,
                "gibberish": 20,
            },
        ),
        "o1-preview-2024-09-12": ModelHeuristics(
            dollars_per_million_input_tokens=15,
            dollars_per_million_output_tokens=60,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=35,
            can_stream=False,
            skills={
                "all": 20,
                "other": 20,
                "opinion": 20,
                "advice": 20,
                "creative writing": 17,
                "mathematics": 20,
                "coding": 20,
                "analysis": 20,
                "entertainment": 20,
                "reasoning": 20,
                "multilingual": 20,
                "summarization": 20,
                "factual": 20,
                "science": 20,
                "small talk": 20,
                "gibberish": 20,
            },
        ),
        "o1-mini-2024-09-12": ModelHeuristics(
            dollars_per_million_input_tokens=3,
            dollars_per_million_output_tokens=12,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=65,
            can_stream=False,
            skills={
                "all": 18,
                "other": 18,
                "opinion": 18,
                "advice": 18,
                "creative writing": 16,
                "mathematics": 18,
                "coding": 18,
                "analysis": 18,
                "entertainment": 18,
                "reasoning": 18,
                "multilingual": 18,
                "summarization": 18,
                "factual": 18,
                "science": 16,
                "small talk": 18,
                "gibberish": 18,
            },
        ),
        "mistral-large-latest": ModelHeuristics(
            dollars_per_million_input_tokens=3,
            dollars_per_million_output_tokens=9,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=35,
            skills={
                "all": 12,
                "other": 18,
                "opinion": 14,
                "advice": 12,
                "creative writing": 10,
                "mathematics": 8,
                "coding": 9,
                "analysis": 11,
                "entertainment": 11,
                "reasoning": 11,
                "multilingual": 11,
                "summarization": 4,
                "factual": 9,
                "science": 12,
                "small talk": 11,
                "gibberish": 20,
            },
        ),
        "codestral-2405": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,
            dollars_per_million_output_tokens=0.6,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=80,
            skills={
                "all": 12,
                "other": 18,
                "opinion": 14,
                "advice": 12,
                "creative writing": 10,
                "mathematics": 9,
                "coding": 13,
                "analysis": 11,
                "entertainment": 11,
                "reasoning": 11,
                "multilingual": 11,
                "summarization": 4,
                "factual": 9,
                "science": 12,
                "small talk": 11,
                "gibberish": 20,
            },
        ),
        "gemini-1.5-pro": ModelHeuristics(
            dollars_per_million_input_tokens=3.5,
            dollars_per_million_output_tokens=10.5,
            tokenizer_type="google",
            tokenizer_name="gemini-1.5-pro",
            tokens_per_second=65,
            skills={
                "all": 12,
                "other": 18,
                "opinion": 14,
                "advice": 14,
                "creative writing": 10,
                "mathematics": 12,
                "coding": 14,
                "analysis": 11,
                "entertainment": 11,
                "reasoning": 11,
                "multilingual": 16,
                "summarization": 4,
                "factual": 9,
                "science": 12,
                "small talk": 11,
                "gibberish": 20,
            },
        ),
        "google/gemma-2-9b-it": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,  # based on fireworks
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=120,
            skills={
                "all": 6,
                "other": 7,
                "opinion": 6,
                "advice": 6,
                "creative writing": 5,
                "mathematics": 4,
                "coding": 4,
                "science": 4,
                "small talk": 7,
                "gibberish": 7,
                "analysis": 4,
                "entertainment": 4,
                "reasoning": 4,
                "multilingual": 2,
                "summarization": 9,
                "factual": 3,
            },
        ),
        "gemini-1.5-flash-8b": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,  # based on fireworks
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=200,  # from ArtificialAnalysis
            skills={
                "all": 12,
                "other": 16,
                "opinion": 12,
                "advice": 10,
                "creative writing": 8,
                "mathematics": 10,
                "coding": 12,
                "analysis": 10,
                "entertainment": 10,
                "reasoning": 10,
                "multilingual": 13,
                "summarization": 6,
                "factual": 10,
                "science": 11,
                "small talk": 11,
                "gibberish": 20,
            },
        ),
        "claude-3-5-sonnet-20240620": ModelHeuristics(
            dollars_per_million_input_tokens=3,
            dollars_per_million_output_tokens=15,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=80,
            skills={
                "advice": 17,
                "all": 14,
                "analysis": 16,
                "coding": 16,
                "creative writing": 20,
                "entertainment": 18,
                "factual": 16,
                "gibberish": 20,
                "mathematics": 8,
                "multilingual": 20,
                "opinion": 17,
                "other": 16,
                "reasoning": 13,
                "science": 12,
                "small talk": 11,
                "summarization": 10,
            },
        ),
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ModelHeuristics(
            dollars_per_million_input_tokens=0.9,  # based on fireworks
            dollars_per_million_output_tokens=0.9,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=90,
            skills={
                "advice": 12,
                "all": 14,
                "analysis": 8,
                "coding": 8,
                "creative writing": 12,
                "entertainment": 10,
                "factual": 10,
                "gibberish": 20,
                "mathematics": 7,
                "multilingual": 20,
                "opinion": 17,
                "other": 16,
                "reasoning": 6,
                "science": 8,
                "small talk": 11,
                "summarization": 10,
            },
        ),
        "microsoft/phi-3-mini-4k-instruct": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,  # based on fireworks
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=130,
            skills={
                "advice": 3,
                "all": 3,
                "analysis": 3,
                "coding": 9,
                "creative writing": 4,
                "entertainment": 3,
                "factual": 3,
                "gibberish": 3,
                "mathematics": 8,
                "multilingual": 3,
                "opinion": 3,
                "other": 3,
                "reasoning": 3,
                "science": 3,
                "small talk": 3,
                "summarization": 3,
            },
        ),
        "phi-3-medium-4k-instruct": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,  # based on fireworks
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=120,
            skills={
                "advice": 3,
                "all": 3,
                "analysis": 3,
                "coding": 9,
                "creative writing": 4,
                "entertainment": 3,
                "factual": 3,
                "gibberish": 3,
                "mathematics": 8,
                "multilingual": 3,
                "opinion": 3,
                "other": 3,
                "reasoning": 3,
                "science": 3,
                "small talk": 3,
                "summarization": 3,
            },
        ),
        "yi-large": ModelHeuristics(
            dollars_per_million_input_tokens=3,  # based on fireworks
            dollars_per_million_output_tokens=3,
            tokenizer_name="gpt-4o",  # approximation
        ),
        "deepseek-coder": ModelHeuristics(
            dollars_per_million_input_tokens=0.9,  # based on fireworks
            dollars_per_million_output_tokens=0.9,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=50,
            skills={
                "advice": 8,
                "all": 8,
                "analysis": 14,
                "coding": 10,
                "creative writing": 6,
                "entertainment": 6,
                "factual": 6,
                "gibberish": 20,
                "mathematics": 12,
                "multilingual": 12,
                "opinion": 10,
                "other": 10,
                "reasoning": 12,
                "science": 14,
                "small talk": 12,
                "summarization": 6,
            },
        ),
        "nemotron-4-340b-instruct": ModelHeuristics(
            dollars_per_million_input_tokens=10,  # rough approximation
            dollars_per_million_output_tokens=10,
            tokenizer_name="gpt-4o",  # approximation
        ),
        "qwen1.5-7b-chat": ModelHeuristics(
            dollars_per_million_input_tokens=3,  # based on Alibaba
            dollars_per_million_output_tokens=3,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=100,
            skills={
                "all": 2,
                "other": 2,
                "opinion": 2,
                "advice": 2,
                "creative writing": 2,
                "mathematics": 3,
                "code": 3,
                "analysis": 3,
                "entertainment": 2,
                "comparison": 2,
                "reasoning": 2,
                "multilingual": 4,
                "summarization": 2,
                "education": 2,
                "factual": 3,
            },
        ),
        "Qwen/Qwen1.5-72B-Chat": ModelHeuristics(
            dollars_per_million_input_tokens=3,  # based on Alibaba
            dollars_per_million_output_tokens=9,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=80,
            skills={
                "advice": 6,
                "all": 10,
                "analysis": 6,
                "coding": 5,
                "creative writing": 9,
                "entertainment": 3,
                "factual": 6,
                "gibberish": 6,
                "mathematics": 6,
                "multilingual": 6,
                "opinion": 10,
                "other": 10,
                "reasoning": 3,
                "science": 13,
                "small talk": 6,
                "summarization": 5,
            },
        ),
        "qwen-max": ModelHeuristics(
            dollars_per_million_input_tokens=10,  # based on Alibaba
            dollars_per_million_output_tokens=10,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=35,
            skills={
                "all": 10,
                "other": 10,
                "opinion": 14,
                "advice": 10,
                "creative writing": 5,
                "mathematics": 7,
                "coding": 8,
                "science": 14,
                "small talk": 14,
                "gibberish": 18,
                "analysis": 14,
                "entertainment": 9,
                "reasoning": 17,
                "multilingual": 4,
                "summarization": 16,
                "factual": 12,
            },
        ),
        "google/gemma-2-27b-it": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=70,
            skills={
                "all": 10,
                "other": 10,
                "opinion": 10,
                "advice": 10,
                "creative writing": 10,
                "mathematics": 8,
                "coding": 6,
                "science": 10,
                "small talk": 10,
                "gibberish": 20,
                "analysis": 6,
                "entertainment": 10,
                "reasoning": 10,
                "multilingual": 10,
                "summarization": 10,
                "factual": 10,
            },
        ),
    }
)

MODEL_HEURISTICS["gpt-4o-mini-2024-07-18"] = MODEL_HEURISTICS["gpt-4o-mini"]
MODEL_HEURISTICS["gemini-1.5-pro-exp-0827"] = MODEL_HEURISTICS["gemini-1.5-pro"]

ALL_MODELS_LOCAL = {
    "qwen/qwen-2.5-72b-instruct",
    "gemini-1.5-pro-exp-0827",
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "llama-3.1-sonar-small-128k-chat",
    "cohere/command-r-08-2024",
    "cohere/command-r-plus",
    "gpt-4o-2024-05-13",
    "databricks/dbrx-instruct",
    "llama-3.1-sonar-large-128k-online",
    "qwen-plus",
    "gemini-1.5-flash-8b",
    "gpt-3.5-turbo-0125",
    "claude-3-5-sonnet-20240620",
    "gpt-4o-mini-2024-07-18",
    "o1-mini-2024-09-12",
    "meta-llama/Llama-3-8b-chat-hf",
    "gpt-4-turbo",
    "meta-llama/llama-3.2-1b-instruct",
    "gpt-4o-mini",
    "Qwen/Qwen1.5-72B-Chat",
    "claude-3-sonnet-20240229",
    "pixtral-12b-2409",
    "gemini-1.5-flash-exp-0827",
    "gemini-1.5-flash-8b-exp-0827",
    "meta-llama/llama-3.1-70b-instruct",
    "open-mixtral-8x7b",
    "llama-3.1-sonar-small-128k-online",
    "meta-llama/Llama-3-70b-chat-hf",
    "deepseek/deepseek-chat",
    "cohere/command-r",
    "google/gemma-2-9b-it",
    "mistral-medium",
    "ai21/jamba-1-5-mini",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "llama-3.1-sonar-huge-128k-online",
    "mistral-large-2402",
    "claude-3-opus-20240229",
    "open-mixtral-8x22b",
    "ai21/jamba-1-5-large",
    "Qwen/Qwen1.5-110B-Chat",
    "qwen-max",
    "claude-3-haiku-20240307",
    "o1-preview-2024-09-12",
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "gpt-4o",
    "Qwen/Qwen2-72B-Instruct",
    "codestral-2405",
    "deepseek-coder",
    "gpt-4o-2024-08-06",
    "google/gemma-2-27b-it",
    "llama-3.1-sonar-large-128k-chat",
    "cohere/command-r-plus-08-2024",
}

MODEL_DESCRIPTIONS = {
    "gpt-4o": (
        "OpenAI's GPT-4o, a good all-around model. It has a larger context size. It's good for most tasks. Pretty fast."
    ),
    "gpt-4o-mini": ("OpenAI's GPT-4o Mini, an inexpensive model. It's decent at most tasks. Pretty fast."),
    "gpt-4-turbo": (
        "OpenAI's GPT-4 Turbo, a high-performance model. Slightly worse than GPT-4o but better than GPT-4o Mini. "
        "Kind of fast."
    ),
    "o1-preview-2024-09-12": (
        "OpenAI's O1, the best model for complex tasks for math, reasoning, and coding. Overkill for nonreasoning and "
        "noncomplex tasks. Slow."
    ),
    "o1-mini-2024-09-12": (
        "OpenAI's O1 Mini, a great model for math, reasoning, and coding. Better for slightly complex tasks than 4o. "
        "Not slow, not fast."
    ),
    "mistral-large-latest": (
        "Mistral's large model, a good all-around model. Not as good as GPT-4o in most tasks. Not slow, not fast."
    ),
    "gemini-1.5-pro": ("Google's Gemini 1.5 Pro, a good model for most tasks. Kind of fast."),
    "gemma-2-9b-it": ("Google's Gemma 9B model, a good model for simple tasks. Pretty fast."),
    "claude-3-5-sonnet-20240620": (
        "Anthropic's Claude 3.5 Sonnet, a good model for most tasks. It's about the same as GPT-4o. Kind of fast."
    ),
    "meta-llama/Meta-Llama-3.1-70B-Instruct": (
        "Meta-Llama 3.1 70B Instruct, a good model for most simple tasks, better than Gemma 9B. Kind of fast."
    ),
    "phi-3-mini-4k-instruct": (
        "Microsoft's Phi 3 Mini 4k Instruct, a very specialized model great for simple coding tasks. Fails at complex "
        "tasks. Very fast."
    ),
    "phi-3-medium-4k-instruct": (
        "Microsoft's Phi 3 Medium 4k Instruct, a very specialized model great for simple-to-medium coding tasks. Fails"
        " at complex tasks. Very fast."
    ),
    "qwen1.5-72b-chat": ("A decent model, especially for Mandarin Chinese. Kind of fast."),
    "qwen-max": ("A good model, especially for Mandarin Chinese. Kind of slow."),
}

PROVIDER_KEY_MAPPING = {
    "alibaba": "ALIBABA_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "anyscale": "ANYSCALE_API_KEY",
    "azure": "AZURE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "google": "GOOGLE_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "togetherai": "TOGETHER_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
}

PROVIDER_MODEL_PATTERNS = {
    re.compile(r"^(chatgpt|gpt-[34]|o1).*$", re.IGNORECASE): "openai",
    re.compile(r"^(gemini|google|gemma|palm).*$", re.IGNORECASE): "google",
    re.compile(r"^(mistral|codestral|pixtral|mixtral).*$", re.IGNORECASE): "mistralai",
    re.compile(r"^(claude|opus).*$", re.IGNORECASE): "anthropic",
    re.compile(r"^(phi|microsoft).*$", re.IGNORECASE): "azure",
    re.compile(r"^(qwen|alibaba).*$", re.IGNORECASE): "alibaba",
    re.compile(r"^(meta|codellama).*$", re.IGNORECASE): "meta",
    re.compile(r"^(ai21|jamba).*$", re.IGNORECASE): "ai21",
}
