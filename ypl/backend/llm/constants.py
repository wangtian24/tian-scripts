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

COSTS_BY_MODEL = {
    "gpt-4o": ModelCost(
        dollars_per_million_input_tokens=5,
        dollars_per_million_output_tokens=15,
        tokenizer_name="gpt-4o",
        tokens_per_second=120,
    ),
    "gpt-4o-mini": ModelCost(
        dollars_per_million_input_tokens=0.15,
        dollars_per_million_output_tokens=0.6,
        tokenizer_name="gpt-4o-mini",
        tokens_per_second=100,
    ),
    "gpt-4o-mini-2024-07-18": ModelCost(
        dollars_per_million_input_tokens=0.15,
        dollars_per_million_output_tokens=0.6,
        tokenizer_name="gpt-4o-mini",
        tokens_per_second=100,
    ),
    "gpt-4-turbo": ModelCost(
        dollars_per_million_input_tokens=10,
        dollars_per_million_output_tokens=30,
        tokenizer_name="gpt-4-turbo",
        tokens_per_second=50,
    ),
    "o1-preview-2024-09-12": ModelCost(
        dollars_per_million_input_tokens=15,
        dollars_per_million_output_tokens=60,
        tokenizer_name="gpt-4o",  # approximation
        tokens_per_second=35,
    ),
    "o1-mini-2024-09-12": ModelCost(
        dollars_per_million_input_tokens=3,
        dollars_per_million_output_tokens=12,
        tokenizer_name="gpt-4o",  # approximation
        tokens_per_second=65,
    ),
    "mistral-large-latest": ModelCost(
        dollars_per_million_input_tokens=3,
        dollars_per_million_output_tokens=9,
        tokenizer_name="gpt-4o",  # approximation
        tokens_per_second=35,
    ),
    "gemini-1.5-pro": ModelCost(
        dollars_per_million_input_tokens=3.5,
        dollars_per_million_output_tokens=10.5,
        tokenizer_type="google",
        tokenizer_name="gemini-1.5-pro",
        tokens_per_second=65,
    ),
    "gemma-2-9b-it": ModelCost(
        dollars_per_million_input_tokens=0.2,  # based on fireworks
        dollars_per_million_output_tokens=0.2,
        tokenizer_name="gpt-4o",  # approximation
        tokens_per_second=100,
    ),
    "claude-3-5-sonnet-20240620": ModelCost(
        dollars_per_million_input_tokens=3,
        dollars_per_million_output_tokens=15,
        tokenizer_name="gpt-4o",  # approximation
        tokens_per_second=80,
    ),
    "meta-llama/Meta-Llama-3.1-70B-Instruct": ModelCost(
        dollars_per_million_input_tokens=0.9,  # based on fireworks
        dollars_per_million_output_tokens=0.9,
        tokenizer_name="gpt-4o",  # approximation
        tokens_per_second=90,
    ),
    "phi-3-mini-4k-instruct": ModelCost(
        dollars_per_million_input_tokens=0.2,  # based on fireworks
        dollars_per_million_output_tokens=0.2,
        tokenizer_name="gpt-4o",  # approximation
        tokens_per_second=130,
    ),
    "phi-3-medium-4k-instruct": ModelCost(
        dollars_per_million_input_tokens=0.2,  # based on fireworks
        dollars_per_million_output_tokens=0.2,
        tokenizer_name="gpt-4o",  # approximation
        tokens_per_second=100,
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
        tokens_per_second=40,
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
        tokens_per_second=100,
    ),
    "qwen1.5-72b-chat": ModelCost(
        dollars_per_million_input_tokens=3,  # based on Alibaba
        dollars_per_million_output_tokens=9,
        tokenizer_name="gpt-4o",  # approximation
        tokens_per_second=100,
    ),
    "qwen-max": ModelCost(
        dollars_per_million_input_tokens=10,  # based on Alibaba
        dollars_per_million_output_tokens=10,
        tokenizer_name="gpt-4o",  # approximation
        tokens_per_second=35,
    ),
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
