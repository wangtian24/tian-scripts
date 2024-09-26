from enum import Enum

from ypl.backend.llm.cost import ModelCost


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

    @classmethod
    def from_string(cls, provider: str) -> "ChatProvider":
        try:
            return cls[provider.upper()]
        except KeyError as e:
            raise ValueError(f"Unsupported provider string: {provider}") from e


# Remember to keep the list of models in sync with
# https://github.com/yupp-ai/sarai-chat/blob/main/lib/llms.tsx#L38
FRONTEND_MODELS_BY_PROVIDER = {
    ChatProvider.OPENAI: [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
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
    ChatProvider.NVIDIA: ["nemotron-4-340b-instruct", "yi-large"],
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
    ChatProvider.HERMES: [
        "hermes-3-llama-3.1-405b-fp8",
    ],
    ChatProvider.OPENAI: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4o-mini-2024-07-18"],
    ChatProvider.MISTRAL: ["mistral-large-latest"],
    ChatProvider.GOOGLE: ["gemini-1.5-pro"],
    ChatProvider.ANTHROPIC: ["claude-3-5-sonnet-20240620"],
    ChatProvider.MICROSOFT: [
        "phi-3-mini-4k-instruct",
    ],
    ChatProvider.DEEPSEEK: [
        "deepseek-coder-v2",
    ],
    ChatProvider.NVIDIA: ["nemotron-4-340b-instruct", "yi-large"],
    ChatProvider.QWEN: ["qwen-max"],
    ChatProvider.ANYSCALE: [
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ],
    ChatProvider.TOGETHER: [
        "gemma-2-9b-it",
        "qwen1.5-72b-chat",
    ],
}

FRONTEND_MODELS = [model for models in FRONTEND_MODELS_BY_PROVIDER.values() for model in models]
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

COSTS_BY_MODEL = {
    "gpt-4o": ModelCost(
        dollars_per_million_input_tokens=5, dollars_per_million_output_tokens=15, tokenizer_name="gpt-4o"
    ),
    "gpt-4o-mini": ModelCost(
        dollars_per_million_input_tokens=0.15, dollars_per_million_output_tokens=0.6, tokenizer_name="gpt-4o-mini"
    ),
    "gpt-4o-mini-2024-07-18": ModelCost(
        dollars_per_million_input_tokens=0.15, dollars_per_million_output_tokens=0.6, tokenizer_name="gpt-4o-mini"
    ),
    "gpt-4-turbo": ModelCost(
        dollars_per_million_input_tokens=10, dollars_per_million_output_tokens=30, tokenizer_name="gpt-4-turbo"
    ),
    "mistral-large-latest": ModelCost(
        dollars_per_million_input_tokens=3,
        dollars_per_million_output_tokens=9,
        tokenizer_name="gpt-4o",  # approximation
    ),
    "gemini-1.5-pro": ModelCost(
        dollars_per_million_input_tokens=3.5,
        dollars_per_million_output_tokens=10.5,
        tokenizer_type="google",
        tokenizer_name="gemini-1.5-pro",
    ),
    "gemma-2-9b-it": ModelCost(
        dollars_per_million_input_tokens=0.2,  # based on fireworks
        dollars_per_million_output_tokens=0.2,
        tokenizer_name="gpt-4o",  # approximation
    ),
    "claude-3-5-sonnet-20240620": ModelCost(
        dollars_per_million_input_tokens=3,
        dollars_per_million_output_tokens=15,
        tokenizer_name="gpt-4o",  # approximation
    ),
    "meta-llama/Meta-Llama-3.1-70B-Instruct": ModelCost(
        dollars_per_million_input_tokens=0.9,  # based on fireworks
        dollars_per_million_output_tokens=0.9,
        tokenizer_name="gpt-4o",  # approximation
    ),
    "phi-3-mini-4k-instruct": ModelCost(
        dollars_per_million_input_tokens=0.2,  # based on fireworks
        dollars_per_million_output_tokens=0.2,
        tokenizer_name="gpt-4o",  # approximation
    ),
    "phi-3-medium-4k-instruct": ModelCost(
        dollars_per_million_input_tokens=0.2,  # based on fireworks
        dollars_per_million_output_tokens=0.2,
        tokenizer_name="gpt-4o",  # approximation
    ),
    "yi-large": ModelCost(
        dollars_per_million_input_tokens=3,  # based on fireworks
        dollars_per_million_output_tokens=3,
        tokenizer_name="gpt-4o",  # approximation
    ),
    "deepseek-coder-v2": ModelCost(
        dollars_per_million_input_tokens=0.9,  # based on fireworks
        dollars_per_million_output_tokens=0.9,
        tokenizer_name="gpt-4o",  # approximation
    ),
    "nemotron-4-340b-instruct": ModelCost(
        dollars_per_million_input_tokens=10,  # rough approximation
        dollars_per_million_output_tokens=10,
        tokenizer_name="gpt-4o",  # approximation
    ),
    "qwen1.5-7b-chat": ModelCost(
        dollars_per_million_input_tokens=3,  # based on Alibaba
        dollars_per_million_output_tokens=3,
        tokenizer_name="gpt-4o",  # approximation
    ),
    "qwen1.5-72b-chat": ModelCost(
        dollars_per_million_input_tokens=3,  # based on Alibaba
        dollars_per_million_output_tokens=9,
        tokenizer_name="gpt-4o",  # approximation
    ),
    "qwen-max": ModelCost(
        dollars_per_million_input_tokens=10,  # based on Alibaba
        dollars_per_million_output_tokens=10,
        tokenizer_name="gpt-4o",  # approximation
    ),
}
