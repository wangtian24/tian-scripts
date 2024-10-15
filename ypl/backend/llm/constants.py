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
MODEL_HEURISTICS: dict[str, ModelHeuristics] = defaultdict(
    lambda: ModelHeuristics(
        dollars_per_million_input_tokens=1,
        dollars_per_million_output_tokens=1,
        tokenizer_name="gpt-4o",
        tokens_per_second=75,
        skills={
            "all": 5,
            "other": 5,
            "opinion": 5,
            "advice": 4,
            "creative writing": 6,
            "math": 4,
            "code": 4,
            "analysis": 6,
            "entertainment": 5,
            "comparison": 5,
            "reasoning": 5,
            "multilingual": 4,
            "summarization": 8,
            "education": 4,
            "factual": 6,
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
                "all": 9,
                "other": 10,
                "opinion": 8,
                "advice": 6,
                "creative writing": 6,
                "math": 7,
                "code": 7,
                "analysis": 8,
                "entertainment": 8,
                "comparison": 8,
                "reasoning": 7,
                "multilingual": 7,
                "summarization": 10,
                "education": 9,
                "factual": 9,
            },
        ),
        "gpt-4o-mini": ModelHeuristics(
            dollars_per_million_input_tokens=0.15,
            dollars_per_million_output_tokens=0.6,
            tokenizer_name="gpt-4o-mini",
            tokens_per_second=100,
            skills={
                "all": 8,
                "other": 10,
                "opinion": 7,
                "advice": 5,
                "creative writing": 5,
                "math": 7,
                "code": 6,
                "analysis": 7,
                "entertainment": 8,
                "comparison": 7,
                "reasoning": 6,
                "multilingual": 6,
                "summarization": 10,
                "education": 8,
                "factual": 8,
            },
        ),
        "gpt-4-turbo": ModelHeuristics(
            dollars_per_million_input_tokens=10,
            dollars_per_million_output_tokens=30,
            tokenizer_name="gpt-4-turbo",
            tokens_per_second=50,
            skills={
                "all": 8,
                "other": 10,
                "opinion": 7,
                "advice": 6,
                "creative writing": 5,
                "math": 7,
                "code": 6,
                "analysis": 7,
                "entertainment": 7,
                "comparison": 7,
                "reasoning": 6,
                "multilingual": 6,
                "summarization": 10,
                "education": 8,
                "factual": 8,
            },
        ),
        "o1-preview-2024-09-12": ModelHeuristics(
            dollars_per_million_input_tokens=15,
            dollars_per_million_output_tokens=60,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=35,
            skills={
                "all": 10,
                "other": 10,
                "opinion": 10,
                "advice": 10,
                "creative writing": 10,
                "math": 10,
                "code": 10,
                "analysis": 10,
                "entertainment": 10,
                "comparison": 10,
                "reasoning": 10,
                "multilingual": 10,
                "summarization": 10,
                "education": 10,
                "factual": 10,
            },
        ),
        "o1-mini-2024-09-12": ModelHeuristics(
            dollars_per_million_input_tokens=3,
            dollars_per_million_output_tokens=12,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=65,
            skills={
                "all": 10,
                "other": 9,
                "opinion": 9,
                "advice": 6,
                "creative writing": 6,
                "math": 9,
                "code": 9,
                "analysis": 9,
                "entertainment": 9,
                "comparison": 9,
                "reasoning": 9,
                "multilingual": 7,
                "summarization": 10,
                "education": 9,
                "factual": 10,
            },
        ),
        "mistral-large-latest": ModelHeuristics(
            dollars_per_million_input_tokens=3,
            dollars_per_million_output_tokens=9,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=35,
            skills={
                "all": 6,
                "other": 8,
                "opinion": 6,
                "advice": 5,
                "creative writing": 4,
                "math": 4,
                "code": 4,
                "analysis": 6,
                "entertainment": 6,
                "comparison": 6,
                "reasoning": 6,
                "multilingual": 4,
                "summarization": 7,
                "education": 5,
                "factual": 6,
            },
        ),
        "codestral-2405": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,
            dollars_per_million_output_tokens=0.6,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=80,
            skills={
                "all": 5,
                "other": 5,
                "opinion": 5,
                "advice": 4,
                "creative writing": 3,
                "math": 4,
                "code": 6,
                "analysis": 5,
                "entertainment": 5,
                "comparison": 5,
                "reasoning": 5,
                "multilingual": 2,
                "summarization": 6,
                "education": 4,
                "factual": 3,
            },
        ),
        "gemini-1.5-pro": ModelHeuristics(
            dollars_per_million_input_tokens=3.5,
            dollars_per_million_output_tokens=10.5,
            tokenizer_type="google",
            tokenizer_name="gemini-1.5-pro",
            tokens_per_second=65,
            skills={
                "all": 8,
                "other": 8,
                "opinion": 8,
                "advice": 5,
                "creative writing": 7,
                "math": 7,
                "code": 5,
                "analysis": 7,
                "entertainment": 7,
                "comparison": 7,
                "reasoning": 6,
                "multilingual": 8,
                "summarization": 9,
                "education": 7,
                "factual": 6,
            },
        ),
        "google/gemma-2-9b-it": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,  # based on fireworks
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=120,
            skills={
                "all": 4,
                "other": 5,
                "opinion": 3,
                "advice": 4,
                "creative writing": 3,
                "math": 3,
                "code": 3,
                "analysis": 3,
                "entertainment": 3,
                "comparison": 3,
                "reasoning": 4,
                "multilingual": 1,
                "summarization": 6,
                "education": 2,
                "factual": 3,
            },
        ),
        "gemini-1.5-flash-8b": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,  # based on fireworks
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=200,  # from ArtificialAnalysis
            skills={
                "all": 4,
                "other": 5,
                "opinion": 3,
                "advice": 4,
                "creative writing": 3,
                "math": 6,
                "code": 6,
                "analysis": 3,
                "entertainment": 3,
                "comparison": 3,
                "reasoning": 5,
                "multilingual": 6,
                "summarization": 6,
                "education": 2,
                "factual": 6,
            },
        ),
        "claude-3-5-sonnet-20240620": ModelHeuristics(
            dollars_per_million_input_tokens=3,
            dollars_per_million_output_tokens=15,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=80,
            skills={
                "all": 7,
                "other": 7,
                "opinion": 7,
                "advice": 10,
                "creative writing": 9,
                "math": 6,
                "code": 8,
                "analysis": 7,
                "entertainment": 8,
                "comparison": 6,
                "reasoning": 6,
                "multilingual": 5,
                "summarization": 9,
                "education": 8,
                "factual": 8,
            },
        ),
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ModelHeuristics(
            dollars_per_million_input_tokens=0.9,  # based on fireworks
            dollars_per_million_output_tokens=0.9,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=90,
            skills={
                "all": 5,
                "other": 5,
                "opinion": 5,
                "advice": 4,
                "creative writing": 6,
                "math": 4,
                "code": 4,
                "analysis": 6,
                "entertainment": 5,
                "comparison": 5,
                "reasoning": 5,
                "multilingual": 4,
                "summarization": 8,
                "education": 4,
                "factual": 7,
            },
        ),
        "microsoft/phi-3-mini-4k-instruct": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,  # based on fireworks
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=130,
            skills={
                "all": 2,
                "other": 2,
                "opinion": 2,
                "advice": 2,
                "creative writing": 2,
                "math": 5,
                "code": 6,
                "analysis": 3,
                "entertainment": 2,
                "comparison": 2,
                "reasoning": 2,
                "multilingual": 0,
                "summarization": 2,
                "education": 2,
                "factual": 3,
            },
        ),
        "phi-3-medium-4k-instruct": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,  # based on fireworks
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=120,
            skills={
                "all": 2,
                "other": 2,
                "opinion": 2,
                "advice": 2,
                "creative writing": 2,
                "math": 6,
                "code": 6,
                "analysis": 4,
                "entertainment": 2,
                "comparison": 2,
                "reasoning": 2,
                "multilingual": 0,
                "summarization": 2,
                "education": 2,
                "factual": 3,
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
                "all": 5,
                "other": 6,
                "opinion": 5,
                "advice": 5,
                "creative writing": 3,
                "math": 8,
                "code": 8,
                "analysis": 5,
                "entertainment": 4,
                "comparison": 3,
                "reasoning": 5,
                "multilingual": 7,
                "summarization": 8,
                "education": 7,
                "factual": 4,
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
                "math": 3,
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
                "all": 3,
                "other": 4,
                "opinion": 4,
                "advice": 3,
                "creative writing": 3,
                "math": 5,
                "code": 5,
                "analysis": 4,
                "entertainment": 3,
                "comparison": 4,
                "reasoning": 4,
                "multilingual": 5,
                "summarization": 5,
                "education": 3,
                "factual": 4,
            },
        ),
        "qwen-max": ModelHeuristics(
            dollars_per_million_input_tokens=10,  # based on Alibaba
            dollars_per_million_output_tokens=10,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=35,
            skills={
                "all": 5,
                "other": 5,
                "opinion": 6,
                "advice": 5,
                "creative writing": 5,
                "math": 5,
                "code": 6,
                "analysis": 6,
                "entertainment": 4,
                "comparison": 5,
                "reasoning": 5,
                "multilingual": 5,
                "summarization": 7,
                "education": 4,
                "factual": 4,
            },
        ),
        "google/gemma-2-27b-it": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=70,
            skills={
                "all": 4,
                "other": 5,
                "opinion": 3,
                "advice": 4,
                "creative writing": 3,
                "math": 4,
                "code": 4,
                "analysis": 3,
                "entertainment": 4,
                "comparison": 4,
                "reasoning": 5,
                "multilingual": 2,
                "summarization": 6,
                "education": 3,
                "factual": 4,
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
    re.compile(r"^(meta|llama|codellama).*$", re.IGNORECASE): "meta",
    re.compile(r"^(ai21|jamba).*$", re.IGNORECASE): "ai21",
}
