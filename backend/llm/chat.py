
from enum import Enum

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr

from backend import prompts


class ChatProvider(Enum):
    OPENAI = 1
    ANTHROPIC = 2
    GOOGLE = 3

    @classmethod
    def from_string(cls, provider: str):
        try:
            return cls[provider.upper()]
        except KeyError as e:
            raise ValueError(f"Unsupported provider string: {provider}") from e


def get_chat_model(provider: ChatProvider | str, model: str, api_key: str) -> BaseChatModel:
    if isinstance(provider, str):
        provider = ChatProvider.from_string(provider)
        print("===", provider)

    chat_llms = {
        ChatProvider.OPENAI: ChatOpenAI,
        ChatProvider.ANTHROPIC: ChatAnthropic,
        ChatProvider.GOOGLE: ChatGoogleGenerativeAI
    }

    chat_llm_cls = chat_llms.get(provider)
    if not chat_llm_cls:
        raise ValueError(f"Unsupported provider: {provider}")

    return chat_llm_cls(api_key=SecretStr(api_key), model=model)

def compare_llm_responses(
        provider: ChatProvider | str, model: str, api_key: str, prompt: str, responses: dict[str, str]
) -> BaseMessage:
    llm = get_chat_model(provider=provider, model=model, api_key=api_key)
    chain = prompts.COMPARE_RESPONSES_PROMPT | llm
    return chain.invoke(input={"prompt": prompt, "responses": responses})
