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
MODELS_BY_PROVIDER = {
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
# Remember to keep the list of models in sync with
# https://github.com/yupp-ai/sarai-chat/blob/main/lib/llms.tsx#L38

MODELS = [model for models in MODELS_BY_PROVIDER.values() for model in models]
