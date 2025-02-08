from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from ypl.backend.llm.constants import ALL_EMBEDDING_MODELS_BY_PROVIDER, ChatProvider
from ypl.backend.llm.model_data_type import ModelInfo


def get_embedding_model(
    info: ModelInfo,
    chat_model_pool: dict[ChatProvider, list[str]] = ALL_EMBEDDING_MODELS_BY_PROVIDER,
    **embedding_kwargs: Any | None,
) -> Embeddings:
    provider, model, api_key = info.provider, info.model, info.api_key

    if isinstance(provider, str):
        provider = ChatProvider.from_string(provider)

    embedding_models = {
        ChatProvider.OPENAI: OpenAIEmbeddings,
    }

    embedding_llm_cls = embedding_models.get(provider)

    if not embedding_llm_cls:
        raise ValueError(f"Unsupported provider: {provider}")

    if model not in chat_model_pool.get(provider, []):
        raise ValueError(f"Unsupported model: {model} for provider: {provider}")

    return embedding_llm_cls(api_key=SecretStr(api_key), model=model, **embedding_kwargs)  # type: ignore
