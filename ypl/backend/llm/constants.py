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
    ChatProvider.GOOGLE: ["gemini-1.5-pro", "gemini-1.5-flash-002"],
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
    ChatProvider.GOOGLE: ["gemini-1.5-pro", "gemini-1.5-flash-002"],
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

ALL_EMBEDDING_MODELS_BY_PROVIDER = {
    ChatProvider.OPENAI: ["text-embedding-ada-002", "text-embedding-3-large", "text-embedding-3-small"]
}

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
    )
)

MODEL_HEURISTICS.update(
    {
        "gpt-4o": ModelHeuristics(
            dollars_per_million_input_tokens=5,
            dollars_per_million_output_tokens=15,
            tokenizer_name="gpt-4o",
            tokens_per_second=120,
        ),
        "gpt-4o-mini": ModelHeuristics(
            dollars_per_million_input_tokens=0.15,
            dollars_per_million_output_tokens=0.6,
            tokenizer_name="gpt-4o-mini",
            tokens_per_second=100,
        ),
        "gpt-4-turbo": ModelHeuristics(
            dollars_per_million_input_tokens=10,
            dollars_per_million_output_tokens=30,
            tokenizer_name="gpt-4-turbo",
            tokens_per_second=50,
        ),
        "o1-preview-2024-09-12": ModelHeuristics(
            dollars_per_million_input_tokens=15,
            dollars_per_million_output_tokens=60,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=35,
            can_stream=False,
        ),
        "o1-mini-2024-09-12": ModelHeuristics(
            dollars_per_million_input_tokens=3,
            dollars_per_million_output_tokens=12,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=65,
            can_stream=True,
        ),
        "mistral-large-latest": ModelHeuristics(
            dollars_per_million_input_tokens=3,
            dollars_per_million_output_tokens=9,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=35,
        ),
        "codestral-2405": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,
            dollars_per_million_output_tokens=0.6,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=80,
        ),
        "gemini-1.5-pro": ModelHeuristics(
            dollars_per_million_input_tokens=3.5,
            dollars_per_million_output_tokens=10.5,
            tokenizer_type="google",
            tokenizer_name="gemini-1.5-pro",
            tokens_per_second=65,
        ),
        "google/gemma-2-9b-it": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,  # based on fireworks
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=120,
        ),
        "gemini-1.5-flash-8b": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,  # based on fireworks
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=200,  # from ArtificialAnalysis
        ),
        "claude-3-5-sonnet-20240620": ModelHeuristics(
            dollars_per_million_input_tokens=3,
            dollars_per_million_output_tokens=15,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=80,
        ),
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ModelHeuristics(
            dollars_per_million_input_tokens=0.9,  # based on fireworks
            dollars_per_million_output_tokens=0.9,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=90,
        ),
        "microsoft/phi-3-mini-4k-instruct": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,  # based on fireworks
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=130,
        ),
        "phi-3-medium-4k-instruct": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,  # based on fireworks
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=120,
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
        ),
        "Qwen/Qwen1.5-72B-Chat": ModelHeuristics(
            dollars_per_million_input_tokens=3,  # based on Alibaba
            dollars_per_million_output_tokens=9,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=80,
        ),
        "qwen-max": ModelHeuristics(
            dollars_per_million_input_tokens=10,  # based on Alibaba
            dollars_per_million_output_tokens=10,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=35,
        ),
        "google/gemma-2-27b-it": ModelHeuristics(
            dollars_per_million_input_tokens=0.2,
            dollars_per_million_output_tokens=0.2,
            tokenizer_name="gpt-4o",  # approximation
            tokens_per_second=70,
        ),
    }
)

MODEL_HEURISTICS["gpt-4o-mini-2024-07-18"] = MODEL_HEURISTICS["gpt-4o-mini"]
MODEL_HEURISTICS["gemini-1.5-pro-exp-0827"] = MODEL_HEURISTICS["gemini-1.5-pro"]

PROVIDER_MODEL_PATTERNS = {
    re.compile(r".*-sonar-.*$", re.IGNORECASE): "perplexity",
    re.compile(r"^(openai|chatgpt|gpt-[34]|o1|o3).*$", re.IGNORECASE): "openai",
    re.compile(r"^(google|gemma|palm|gemini|models/gemini).*$", re.IGNORECASE): "google",
    re.compile(
        r"^(mistral|codestral|pixtral|mixtral|ministral|open-mistral|open-mixtral).*$", re.IGNORECASE
    ): "mistralai",
    re.compile(r"^(claude|opus).*$", re.IGNORECASE): "anthropic",
    re.compile(r"^(phi|microsoft).*$", re.IGNORECASE): "azure",
    re.compile(r"^(qwen|alibaba).*$", re.IGNORECASE): "alibaba",
    re.compile(r"^(meta|codellama|llama).*$", re.IGNORECASE): "meta",
    re.compile(r"^(nousresearch).*$", re.IGNORECASE): "nousresearch",
    re.compile(r"^(ai21|jamba).*$", re.IGNORECASE): "ai21",
    re.compile(r"^(databricks).*$", re.IGNORECASE): "databricks",
    re.compile(r"^(deepseek).*$", re.IGNORECASE): "deepseek",
    re.compile(r"^(amazon/).*$", re.IGNORECASE): "amazon",
    re.compile(r"^(cohere/).*$", re.IGNORECASE): "cohere",
    re.compile(r"^(nvidia).*$", re.IGNORECASE): "nvidia",
    re.compile(r"^(x-ai).*$", re.IGNORECASE): "x-ai",
    re.compile(r"^(gryphe).*$", re.IGNORECASE): "gryphe",
    re.compile(r"^(sambanova).*$", re.IGNORECASE): "sambanova",
    re.compile(r"^(falai).*$", re.IGNORECASE): "falai",
}


IMAGE_CATEGORY = "image"
PDF_CATEGORY = "pdf"
ONLINE_CATEGORY = "online"
OFFLINE_CATEGORY = "offline"
IMAGE_GEN_CATEGORY = "Image Generation"
CODING_CATEGORY = "Coding"
