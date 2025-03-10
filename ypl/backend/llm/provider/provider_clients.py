import logging
import os
from typing import Any

from cachetools.func import ttl_cache
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr
from sqlmodel import Session, select
from ypl.backend.config import settings
from ypl.backend.db import get_engine
from ypl.backend.llm.constants import ChatProvider
from ypl.backend.llm.model_data_type import ModelInfo
from ypl.backend.llm.provider.google_grounded_gemini import GoogleGroundedGemini
from ypl.backend.llm.provider.google_grounded_vertex_ai import GroundedVertexAI
from ypl.backend.llm.provider.image_gen_models import DallEChatModel, FalAIImageGenModel
from ypl.backend.llm.provider.perplexity import CustomChatPerplexity
from ypl.backend.llm.vendor_langchain_adapter import GeminiLangChainAdapter, OpenAILangChainAdapter
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
    "Yapp": "YAPP_API_KEY",
    "Yapp Temporary": "YAPP_TMP_API_KEY",
}
PROVIDER_KWARGS = {
    "OpenRouter": {"extra_body": {"transforms": ["middle-out"]}},
}


@ttl_cache(ttl=600)  # 600 seconds = 10 minutes
def load_models_with_providers(include_all_models: bool = False) -> dict[str, tuple[LanguageModel, Provider]]:
    """
    Load all active language models with their provider information from the database.
    Returns a map from model name (not internal_name!!) to (model, provider) tuple.
    """
    with Session(get_engine()) as session:
        conditions = [
            LanguageModel.deleted_at.is_(None),  # type: ignore
            Provider.is_active.is_(True),  # type: ignore
            Provider.deleted_at.is_(None),  # type: ignore
        ]

        if not include_all_models:
            conditions.append(LanguageModel.status == LanguageModelStatusEnum.ACTIVE)

        query = (
            select(LanguageModel, Provider)
            .join(Provider, LanguageModel.provider_id == Provider.provider_id)  # type: ignore
            .where(*conditions)
        )

        results = session.exec(query)
        models_with_providers = results.all()

        return {model.name: (model, provider) for model, provider in models_with_providers}


def get_model_provider_tuple(
    internal_name: str | None = None, name: str | None = None, include_all_models: bool = False
) -> tuple[LanguageModel, Provider] | None:
    """
    Look up the (model, provider) tuple for a given model internal_name of (full) name.
    Cache results for 10 minutes using ttl_cache.
    Returns None if the model is not found.
    """
    model_provider_map_by_name = load_models_with_providers(include_all_models)
    if name is not None:
        return model_provider_map_by_name.get(name)
    elif internal_name is not None:
        model_provider_map_by_internal_name = {
            model.internal_name: (model, provider) for model, provider in model_provider_map_by_name.values()
        }
        return model_provider_map_by_internal_name.get(internal_name)
    else:
        raise ValueError("Either internal_name or name must be provided")


# TODO(bhanu) - add provider to client mapping in DB and remove switch cases (pre-work API key storage)
async def get_provider_client(
    internal_name: str | None = None, name: str | None = None, include_all_models: bool = False, **func_kwargs: Any
) -> BaseChatModel:
    """
    Initialize a LangChain client based on model name.
    Uses cached model and provider details to configure appropriate client.

    Args:
        internal_name: Name of the model to initialize client for, to be deprecated as this is not globally unique
        name: Name of the model, in the form of provider_name/model_name, from the 'name' field in DB
        include_all: If True, include all models, even if they are not active
    """
    if name is not None:
        model_provider = get_model_provider_tuple(name=name, include_all_models=include_all_models)
    elif internal_name is not None:
        model_provider = get_model_provider_tuple(internal_name=internal_name, include_all_models=include_all_models)
    if not model_provider:
        raise ValueError("Either internal_name or name must be provided")

    model, provider = model_provider
    model_db_parameters = model.parameters
    model_db_kwargs = {}
    # some model might want to specify a different port on the provider, this is mostly for internal testing providers
    provider_port = None
    if model_db_parameters:
        if "kwargs" in model_db_parameters:
            model_db_kwargs.update(model_db_parameters["kwargs"])
        if "port" in model_db_parameters:
            provider_port = model_db_parameters["port"]

    # combine extra args for provider and model, the latter overrides the former, the order is
    #    hardcoded < those from DB < those from function caller
    combined_kwargs = {**PROVIDER_KWARGS.get(provider.name, {}), **model_db_kwargs, **func_kwargs}

    # Split the model name into base and variant. The variant is not directly used, variant-specific parameters
    # are stored in the LanguageModel.parameters field.
    # Use "::" as a internal delimiter for model name and variant, the double colon is necessary as there are some
    # models that have ":" in their name, we don't want thtem to be split.
    internal_name = model.internal_name.split("::")[0]
    # model_variant = model.internal_name.split(":")[1] if ":" in model.internal_name else None

    match provider.name:
        case "VertexAI":
            # experimental and preview models are not available in all AZs resulting in 404 model-not-found.
            # to avoid this we are sending them to the default location of us-central1.
            # Gemini 1.5 & 2 are available in multiple AZs (including us-east4), we are not specifying project and LOC.
            # Ref - https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
            if "exp" in internal_name or "preview" in internal_name:
                return ChatVertexAI(
                    model_name=internal_name,
                    project=settings.GCP_PROJECT_ID,
                    location=settings.GCP_REGION_GEMINI_2,
                    **combined_kwargs,
                )
            else:
                return ChatVertexAI(model_name=internal_name, **combined_kwargs)

        case "FalAI":
            return FalAIImageGenModel(model_name=model.internal_name, **combined_kwargs)

        case "GroundedVertexAI":
            if "exp" in internal_name or "preview" in internal_name:
                return GroundedVertexAI(
                    model=internal_name.replace("-online", ""),
                    project=settings.GCP_PROJECT_ID,
                    location=settings.GCP_REGION_GEMINI_2,
                    **combined_kwargs,
                )
            else:
                return GroundedVertexAI(model=internal_name.replace("-online", ""), **combined_kwargs)
        case "Google":
            return ChatGoogleGenerativeAI(
                model=internal_name,
                api_key=SecretStr(os.getenv("GOOGLE_API_KEY", "")),
                **combined_kwargs,
            )

        case "GoogleGrounded":
            return GoogleGroundedGemini(  # type: ignore[call-arg]
                model_info=ModelInfo(
                    provider="GoogleGrounded",
                    model=internal_name.replace("-online", ""),  # TODO: make more robust
                    api_key=settings.GOOGLE_API_KEY,
                ),
                model_config_=dict(
                    project_id=settings.GCP_PROJECT_ID,
                    region=settings.GCP_REGION_GEMINI_2,
                    temperature=0.0,
                    **combined_kwargs,
                ),
            )

        case "OpenAI":
            return ChatOpenAI(
                model=internal_name, api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")), **combined_kwargs
            )

        case "OpenAIDallE":
            return DallEChatModel(openai_api_key=os.getenv("OPENAI_API_KEY", ""), **combined_kwargs)

        case "Anthropic":
            return ChatAnthropic(
                model_name=internal_name, api_key=SecretStr(os.getenv("ANTHROPIC_API_KEY", "")), **combined_kwargs
            )

        case "Mistral AI":
            return ChatMistralAI(
                model_name=internal_name, api_key=SecretStr(os.getenv("MISTRAL_API_KEY", "")), **combined_kwargs
            )
        case "Hugging Face":
            llm = HuggingFaceEndpoint(
                model=internal_name, huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY", ""), **combined_kwargs
            )
            return ChatHuggingFace(llm=llm)
        case "Perplexity":
            return CustomChatPerplexity(
                model=internal_name, api_key=os.getenv("PERPLEXITY_API_KEY", ""), **combined_kwargs
            )

        case _:
            # for all other providers, assume OpenAI-compatible API.
            if provider.api_key_env_name is None and provider.name not in API_KEY_MAP:
                logging.warning(f"No API key env var name for provider [{provider.name}]")
            return ChatOpenAI(
                model=internal_name,
                api_key=SecretStr(os.getenv(provider.api_key_env_name or API_KEY_MAP[provider.name], "")),
                base_url=merge_base_url_with_port(provider.base_api_url, provider_port),
                **combined_kwargs,
            )


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


# -----------------------------------
# TODO(Tian): Below are special provider client getters for various internal uses, we should properly
# support them with get_provider_client() function instead of creating these specialized functions
# that might go out of sync with our main logic.


class InternalLLMParams(BaseModel):
    max_tokens: int
    temperature_times_100: int

    class Config:
        frozen = True  # Make the class immutable and hashable


# Cached specialized LLMs from (name, params) to LLM client objects
COMMON_INTERNAL_LLMS: dict[tuple[str, InternalLLMParams], BaseChatModel] = {}


def get_gemini_2_flash_llm(max_tokens: int) -> GeminiLangChainAdapter:
    return GeminiLangChainAdapter(
        model_info=ModelInfo(
            provider=ChatProvider.GOOGLE,
            model="gemini-2.0-flash-exp",
            api_key=settings.GOOGLE_API_KEY,
        ),
        model_config_=dict(
            project_id=settings.GCP_PROJECT_ID,
            region=settings.GCP_REGION_GEMINI_2,
            temperature=0.0,
            max_output_tokens=max_tokens,
            top_k=1,
        ),
    )


def get_gemini_15_flash_llm(max_tokens: int) -> GeminiLangChainAdapter:
    return GeminiLangChainAdapter(
        model_info=ModelInfo(
            provider=ChatProvider.GOOGLE,
            model="gemini-1.5-flash-002",
            api_key=settings.GOOGLE_API_KEY,
        ),
        model_config_=dict(
            project_id=settings.GCP_PROJECT_ID,
            region=settings.GCP_REGION,
            temperature=0.0,
            max_output_tokens=max_tokens,
            top_k=1,
        ),
    )


def _get_gpt_llm(model: str, max_tokens: int) -> OpenAILangChainAdapter:
    return OpenAILangChainAdapter(
        model_info=ModelInfo(
            provider=ChatProvider.OPENAI,
            model=model,
            api_key=settings.OPENAI_API_KEY,
        ),
        model_config_=dict(
            temperature=0.0,
            max_tokens=max_tokens,
        ),
    )


def get_gpt_4o_llm(max_tokens: int) -> OpenAILangChainAdapter:
    return _get_gpt_llm("gpt-4o", max_tokens)


def get_gpt_4o_mini_llm(max_tokens: int) -> OpenAILangChainAdapter:
    return _get_gpt_llm("gpt-4o-mini", max_tokens)


# TODO(Tian): for backward compatibility, but we probably should remove these soon.
SPECIAL_INTERNAL_LLM_GETTERS = {
    "gpt-4o": get_gpt_4o_llm,
    "gpt-4o-mini": get_gpt_4o_mini_llm,
    "gemini-1.5-flash-002": get_gemini_15_flash_llm,
    "gemini-2.0-flash-exp": get_gemini_2_flash_llm,
}


async def get_internal_provider_client(model_name: str, max_tokens: int, temperature: float = 0.0) -> BaseChatModel:
    params = InternalLLMParams(temperature_times_100=int(temperature * 100), max_tokens=max_tokens)
    if (model_name, params) not in COMMON_INTERNAL_LLMS:
        if model_name in SPECIAL_INTERNAL_LLM_GETTERS:
            COMMON_INTERNAL_LLMS[(model_name, params)] = SPECIAL_INTERNAL_LLM_GETTERS[model_name](max_tokens)
        else:
            COMMON_INTERNAL_LLMS[(model_name, params)] = await get_provider_client(
                internal_name=model_name, include_all_models=True, temperature=temperature, max_tokens=max_tokens
            )
    return COMMON_INTERNAL_LLMS[(model_name, params)]
