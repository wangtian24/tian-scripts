import os

from cachetools.func import ttl_cache
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from sqlalchemy import select
from sqlmodel import Session
from ypl.backend.db import get_engine
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, Provider

load_dotenv()  # Load environment variables from .env file


# TODO(bhanu) - this should auto refresh every 10 minutes and at startup
# TODO(bhanu) - test new model info is hot reloaded in under 10mins (refresh interval)
@ttl_cache(ttl=600)  # 600 seconds = 10 minutes
def load_active_models_with_providers() -> dict[str, tuple[LanguageModel, Provider]]:
    """Load all active language models with their provider information from the database."""
    with Session(get_engine()) as session:
        query = (
            select(LanguageModel, Provider)
            .join(Provider, LanguageModel.provider_id == Provider.provider_id)  # type: ignore
            .where(
                LanguageModel.status == LanguageModelStatusEnum.ACTIVE,  # type: ignore
                Provider.is_active == True,  # type: ignore # noqa: E712
            )
        )

        results = session.exec(query)  # type: ignore[call-overload]
        active_models_with_providers = results.all()

        return {model.internal_name: (model, provider) for model, provider in active_models_with_providers}


def get_model_provider_tuple(model_name: str) -> tuple[LanguageModel, Provider] | None:
    """
    Look up the (model, provider) tuple for a given model name.
    Cache results for 10 minutes using ttl_cache.
    Returns None if the model is not found.
    """
    model_provider_map = load_active_models_with_providers()
    return model_provider_map.get(model_name)


# TODO(bhanu) - add provider to client mapping in DB and remove switch cases (pre-work API key storage)
# TODO(bhanu) - use keys from ypl/backend/config.py
async def get_provider_client(model_name: str) -> BaseChatModel:
    """
    Initialize a LangChain client based on model name.
    Uses cached model and provider details to configure appropriate client.

    Args:
        model_name: Name of the model to initialize client for
    """
    model_provider = get_model_provider_tuple(model_name)
    if not model_provider:
        raise ValueError(f"No model-provider configuration found for: {model_name}")

    model, provider = model_provider

    match provider.name:
        case "Google":
            return ChatGoogleGenerativeAI(
                model=model.internal_name,
                api_key=SecretStr(os.getenv("GOOGLE_API_KEY", "")),
            )

        case "OpenAI":
            return ChatOpenAI(model=model.internal_name, api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")))

        case "Anthropic":
            return ChatAnthropic(  # type: ignore[call-arg]
                model_name=model.internal_name, api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY", ""))
            )

        case "Mistral AI":
            return ChatMistralAI(model_name=model.internal_name, api_key=SecretStr(os.getenv("MISTRAL_API_KEY", "")))
        # TODO(bhanu) - the current API key is throwing 403
        case "Hugging Face":
            llm = HuggingFaceEndpoint(
                model=model.internal_name, huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY", "")
            )
            return ChatHuggingFace(llm=llm)

        case provider_name if provider_name in [
            "OpenRouter",
            "Together AI",
            "Perplexity",
            "Alibaba",
            "DeepSeek",
            "Groq",
            "Cerebras",
            "Anyscale",
        ]:
            api_key_map = {
                "OpenRouter": "OPENROUTER_API_KEY",
                "Together AI": "TOGETHER_API_KEY",
                "Perplexity": "PERPLEXITY_API_KEY",
                "Alibaba": "ALIBABA_API_KEY",
                "DeepSeek": "DEEPSEEK_API_KEY",
                "Groq": "GROQ_API_KEY",
                "Cerebras": "CEREBRAS_API_KEY",
                "Anyscale": "ANYSCALE_API_KEY",
            }
            return ChatOpenAI(
                model=model.internal_name,
                api_key=SecretStr(os.getenv(api_key_map[provider_name], "")),
                base_url=provider.base_api_url,
            )
        # TODO(bhanu) - review inactive providers in DB - Azure, Nvidia, Fireworks
        case _:
            raise ValueError(f"Unsupported provider: {provider.name}")


async def get_language_model(model_name: str) -> LanguageModel:
    """Get the language model details from the database.

    Args:
        model_name: The name of the model to retrieve

    Returns:
        LanguageModel: The language model details

    Raises:
        ValueError: If model is not found
    """
    result = get_model_provider_tuple(model_name)
    if result is None:
        raise ValueError(f"No language model found for model name: {model_name}")

    language_model, _ = result
    return language_model
