from enum import Enum


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
    ChatProvider.GOOGLE: [
        "gemini-1.5-pro",
        "gemma-2-9b-it",
    ],
    ChatProvider.ANTHROPIC: ["claude-3-5-sonnet-20240620"],
    ChatProvider.META: [
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ],
    ChatProvider.MICROSOFT: [
        "phi-3-mini-4k-instruct",
    ],
    ChatProvider.ZERO_ONE: [
        "yi-large",
    ],
    ChatProvider.DEEPSEEK: [
        "deepseek-coder-v2",
    ],
    ChatProvider.NVIDIA: [
        "nemotron-4-340b-instruct",
    ],
    ChatProvider.QWEN: [
        "qwen1.5-7b-chat",
        "qwen-max",
    ],
}

ALL_MODELS_BY_PROVIDER = {
    ChatProvider.OPENAI: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4o-mini-2024-07-18"],
    ChatProvider.MISTRAL: ["mistral-large-latest"],
    ChatProvider.GOOGLE: [
        "gemini-1.5-pro",
        "gemma-2-9b-it",
    ],
    ChatProvider.ANTHROPIC: ["claude-3-5-sonnet-20240620"],
    ChatProvider.META: [
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ],
    ChatProvider.MICROSOFT: [
        "phi-3-mini-4k-instruct",
        "phi-3-medium-4k-instruct",
    ],
    ChatProvider.ZERO_ONE: [
        "yi-large",
    ],
    ChatProvider.DEEPSEEK: [
        "deepseek-coder-v2",
    ],
    ChatProvider.NVIDIA: [
        "nemotron-4-340b-instruct",
    ],
    ChatProvider.QWEN: [
        "qwen1.5-7b-chat",
        "qwen-max",
    ],
    ChatProvider.HERMES: [
        "hermes-3-llama-3.1-405b-fp8",
    ],
}

FRONTEND_MODELS = [model for models in FRONTEND_MODELS_BY_PROVIDER.values() for model in models]

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
