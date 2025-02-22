import os

from cachetools.func import ttl_cache
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from sqlmodel import Session, select
from ypl.backend.config import settings
from ypl.backend.db import get_engine
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.provider.google_grounded_gemini import GoogleGroundedGemini
from ypl.backend.llm.provider.perplexity import CustomChatPerplexity
from ypl.backend.utils.utils import merge_base_url_with_port
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, Provider

load_dotenv()  # Load environment variables from .env file

API_KEY_MAP = {
    "OpenRouter": "OPENROUTER_API_KEY",
    "Together AI": "TOGETHER_API_KEY",
    "Perplexity": "PERPLEXITY_API_KEY",
    "Alibaba": "ALIBABA_API_KEY",
    "DeepSeek": "DEEPSEEK_API_KEY",
    "Groq": "GROQ_API_KEY",
    "Cerebras": "CEREBRAS_API_KEY",
    "Anyscale": "ANYSCALE_API_KEY",
    "Yapp Temporary": "YAPP_TMP_API_KEY",
}
PROVIDER_KWARGS = {
    "OpenRouter": {"extra_body": {"transforms": ["middle-out"]}},
}


# TODO(bhanu) - this should auto refresh every 10 minutes and at startup
# TODO(bhanu) - test new model info is hot reloaded in under 10mins (refresh interval)
@ttl_cache(ttl=600)  # 600 seconds = 10 minutes
def load_active_models_with_providers(include_all_models: bool = False) -> dict[str, tuple[LanguageModel, Provider]]:
    """Load all active language models with their provider information from the database."""
    with Session(get_engine()) as session:
        query = (
            select(LanguageModel, Provider)
            .join(Provider, LanguageModel.provider_id == Provider.provider_id)  # type: ignore
            .where(
                LanguageModel.status == LanguageModelStatusEnum.ACTIVE if not include_all_models else True,
                Provider.is_active == True,  # noqa: E712
            )
        )

        results = session.exec(query)
        active_models_with_providers = results.all()

        return {model.internal_name: (model, provider) for model, provider in active_models_with_providers}


# TODO: Ralph's comment: probably want to standardize the provider name using `standardize_provider_name`
def get_model_provider_tuple(
    model_name: str, include_all_models: bool = False
) -> tuple[LanguageModel, Provider] | None:
    """
    Look up the (model, provider) tuple for a given model name.
    Cache results for 10 minutes using ttl_cache.
    Returns None if the model is not found.
    """
    model_provider_map = load_active_models_with_providers(include_all_models)
    return model_provider_map.get(model_name)


# TODO(bhanu) - add provider to client mapping in DB and remove switch cases (pre-work API key storage)
# TODO(bhanu) - use keys from ypl/backend/config.py
async def get_provider_client(model_name: str, include_all_models: bool = False) -> BaseChatModel:
    """
    Initialize a LangChain client based on model name.
    Uses cached model and provider details to configure appropriate client.

    Args:
        model_name: Name of the model to initialize client for
        include_all: If True, include all models, even if they are not active
    """
    model_provider = get_model_provider_tuple(model_name, include_all_models)
    if not model_provider:
        raise ValueError(f"No model-provider configuration found for: {model_name}")

    model, provider = model_provider
    model_parameters = model.parameters
    model_kwargs = {}
    # some model might want to specify a different port on the provider, this is mostly for internal testing providers
    provider_port = None
    if model_parameters:
        if "kwargs" in model_parameters:
            model_kwargs.update(model_parameters["kwargs"])
        if "port" in model_parameters:
            provider_port = model_parameters["port"]

    # combine extra args for provider and model.
    kwargs = {**PROVIDER_KWARGS.get(provider.name, {}), **model_kwargs}

    # Split the model name into base and variant. The variant is not directly used, variant-specific parameters
    # are stored in the LanguageModel.parameters field.
    model_name = model.internal_name.split(":")[0]
    # model_variant = model.internal_name.split(":")[1] if ":" in model.internal_name else None

    match provider.name:
        case "VertexAI":
            return ChatVertexAI(model_name=model_name, **kwargs)
        case "Google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                api_key=SecretStr(os.getenv("GOOGLE_API_KEY", "")),
                **model_kwargs,
            )

        case "GoogleGrounded":
            return GoogleGroundedGemini(  # type: ignore[call-arg]
                model_info=ModelInfo(
                    provider="GoogleGrounded",
                    model=model_name.replace("-online", ""),  # TODO: make more robust
                    api_key=settings.GOOGLE_API_KEY,
                ),
                model_config_=dict(
                    project_id=settings.GCP_PROJECT_ID,
                    region=settings.GCP_REGION_GEMINI_2,
                    temperature=0.0,
                    **model_kwargs,
                ),
            )

        case "OpenAI":
            return ChatOpenAI(model=model_name, api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")), **kwargs)

        case "Anthropic":
            return ChatAnthropic(model_name=model_name, api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY", "")), **kwargs)

        case "Mistral AI":
            return ChatMistralAI(model_name=model_name, api_key=SecretStr(os.getenv("MISTRAL_API_KEY", "")), **kwargs)
        # TODO(bhanu) - the current API key is throwing 403
        case "Hugging Face":
            llm = HuggingFaceEndpoint(
                model=model_name, huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY", ""), **kwargs
            )
            return ChatHuggingFace(llm=llm)
        case "Perplexity":
            return CustomChatPerplexity(model=model_name, api_key=os.getenv("PERPLEXITY_API_KEY", ""), **kwargs)

        case provider_name if provider_name in [
            "Alibaba",
            "Anyscale",
            "Cerebras",
            "DeepSeek",
            "Groq",
            "OpenRouter",
            "Sambanova",
            "Together AI",
            "Yapp Temporary",  # TODO(tian): add 'Yapp' provider here when we added it to the DB later
        ]:
            return ChatOpenAI(
                model=model_name,
                api_key=SecretStr(os.getenv(provider.api_key_env_name or API_KEY_MAP[provider_name], "")),
                base_url=merge_base_url_with_port(provider.base_api_url, provider_port),
                **kwargs,
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
