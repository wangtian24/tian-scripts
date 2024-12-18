import hashlib
import json
import pathlib
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from pydantic.v1 import SecretStr

from ypl.backend.config import settings
from ypl.backend.llm.chat import ModelInfo
from ypl.backend.llm.constants import ALL_EMBEDDING_MODELS_BY_PROVIDER, ChatProvider


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


def cached_get_database(
    documents: list[Document],
    embedding_model: Embeddings,
    cache_name: str | None = None,
) -> FAISS:
    """
    Returns a flatfile vector database with the given documents, using `embedding_model` to create the embeddings if
    specified. If `cache_name` is provided, the database will be cached using the name; otherwise, it will be
    automatically generated using the hash of the documents.
    """

    def hash_sha512(documents: list[Document]) -> str:
        return hashlib.sha512(json.dumps([doc.dict() for doc in documents]).encode()).hexdigest()

    cache_path = pathlib.Path(settings.CACHE_DIR) / f"{cache_name or hash_sha512(documents)}.faiss.d"

    if cache_path.exists():
        return FAISS.load_local(str(cache_path), embeddings=embedding_model, allow_dangerous_deserialization=True)

    cache_path.mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local(str(cache_path))

    return db
