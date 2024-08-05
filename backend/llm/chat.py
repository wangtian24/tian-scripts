from typing import Any

import torch
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from nltk.tokenize import sent_tokenize, word_tokenize
from pydantic.v1 import SecretStr
from sentence_transformers import SentenceTransformer

from backend import prompts
from backend.llm.constants import MODELS_BY_PROVIDER, ChatProvider
from backend.llm.utils import combine_short_sentences

DEFAULT_HIGH_SIM_THRESHOLD = 0.825
DEFAULT_UNIQUENESS_THRESHOLD = 0.75


def get_chat_model(provider: ChatProvider | str, model: str, api_key: str) -> BaseChatModel:
    if isinstance(provider, str):
        provider = ChatProvider.from_string(provider)

    chat_llms = {
        ChatProvider.OPENAI: ChatOpenAI,
        ChatProvider.ANTHROPIC: ChatAnthropic,
        ChatProvider.GOOGLE: ChatGoogleGenerativeAI,
        ChatProvider.MISTRAL: ChatMistralAI,
    }

    chat_llm_cls = chat_llms.get(provider)
    if not chat_llm_cls:
        raise ValueError(f"Unsupported provider: {provider}")

    if model not in MODELS_BY_PROVIDER.get(provider, []):
        raise ValueError(f"Unsupported model: {model} for provider: {provider}")

    return chat_llm_cls(api_key=SecretStr(api_key), model=model)  # type: ignore


def compare_llm_responses(
    provider: ChatProvider | str, model: str, api_key: str, prompt: str, responses: dict[str, str]
) -> BaseMessage:
    llm = get_chat_model(provider=provider, model=model, api_key=api_key)
    chain = prompts.COMPARE_RESPONSES_PROMPT | llm
    return chain.invoke(input={"prompt": prompt, "responses": responses})


def highlight_llm_similarities(
    provider: ChatProvider | str, model: str, api_key: str, responses: dict[str, str]
) -> BaseMessage:
    llm = get_chat_model(provider=provider, model=model, api_key=api_key)
    chain = prompts.HIGHLIGHT_SIMILARITIES_PROMPT | llm
    return chain.invoke(input={"prompt": "None", "responses": responses})


def highlight_llm_similarities_with_embeddings(
    response_a: str,
    response_b: str,
    high_sim_threshold: float = DEFAULT_HIGH_SIM_THRESHOLD,
    uniqueness_threshold: float = DEFAULT_UNIQUENESS_THRESHOLD,
) -> dict[str, list[str] | list[dict[str, Any]]]:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences_a = combine_short_sentences(sent_tokenize(response_a))
    sentences_b = combine_short_sentences(sent_tokenize(response_b))

    embeddings_a = model.encode(sentences_a, convert_to_tensor=True)
    embeddings_b = model.encode(sentences_b, convert_to_tensor=True)

    similarities = model.similarity(embeddings_a, embeddings_b)  # type: ignore

    high_similarity_pairs = []
    unique_sentences_a = []
    unique_sentences_b = []

    # Find high-similarity pairs
    for i, row in enumerate(similarities):
        for j, sim in enumerate(row):
            if sim >= high_sim_threshold:
                high_similarity_pairs.append(
                    {
                        "sentence_a": sentences_a[i],
                        "sentence_b": sentences_b[j],
                        "similarity": round(sim.item(), 4),
                    }
                )

    # Find unique sentences
    max_similarities_a = torch.max(similarities, dim=1)
    max_similarities_b = torch.max(similarities, dim=0)

    for i, max_sim in enumerate(max_similarities_a.values):
        if max_sim < uniqueness_threshold:
            unique_sentences_a.append(sentences_a[i])

    for j, max_sim in enumerate(max_similarities_b.values):
        if max_sim < uniqueness_threshold:
            unique_sentences_b.append(sentences_b[j])

    return {
        "high_similarity_pairs": high_similarity_pairs,
        "unique_sentences_a": unique_sentences_a,
        "unique_sentences_b": unique_sentences_b,
    }


def prompt_difficulty(
    prompt: str,
    responses: list[str],
    embedding_similarity_weight: float = 0.5,
    structure_similarity_weight: float = 0.3,
    content_similarity_weight: float = 0.2,
) -> dict[str, Any]:
    # Embedding similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(responses, convert_to_tensor=True)
    similarity_matrix = model.similarity(embeddings, embeddings)  # type: ignore
    n = len(responses)
    avg_embedding_similarity = torch.mean(similarity_matrix[torch.triu_indices(n, n, offset=1)])

    # Structure similarity
    sentence_counts = torch.tensor([len(sent_tokenize(response)) for response in responses])
    avg_sentence_count = torch.mean(sentence_counts.float())
    sentence_count_variance = torch.var(sentence_counts.float())
    structure_similarity = 1 - (sentence_count_variance / avg_sentence_count)
    coeff_variation = torch.sqrt(sentence_count_variance) / avg_sentence_count
    structure_similarity = 1 / (1 + coeff_variation)

    # Content similarity
    word_sets = [set(word_tokenize(response.lower())) for response in responses]
    common_words = set.intersection(*word_sets)
    total_words = set.union(*word_sets)
    content_similarity = len(common_words) / len(total_words)

    prompt_difficulty = (
        (1 - avg_embedding_similarity) * embedding_similarity_weight
        + (1 - structure_similarity) * structure_similarity_weight
        + (1 - content_similarity) * content_similarity_weight
    )

    return {
        "prompt_difficulty": prompt_difficulty.item(),
        "embedding_similarity": avg_embedding_similarity.item(),
        "structure_similarity": structure_similarity.item(),
        "content_similarity": content_similarity,
    }


def prompt_difficulty_by_llm(provider: ChatProvider | str, model: str, api_key: str, prompt: str) -> BaseMessage:
    llm = get_chat_model(provider=provider, model=model, api_key=api_key)
    chain = prompts.PROMPT_DIFFICULTY_PROMPT | llm
    return chain.invoke(input={"prompt": prompt})


def prompt_difficulty_by_llm_with_responses(
    provider: ChatProvider | str, model: str, api_key: str, prompt: str, responses: dict[str, str]
) -> BaseMessage:
    llm = get_chat_model(provider=provider, model=model, api_key=api_key)
    chain = prompts.PROMPT_DIFFICULTY_WITH_RESPONSES_PROMPT | llm
    return chain.invoke(input={"prompt": prompt, "responses": responses})
