import logging
import os
import uuid
from typing import Literal

import aiohttp
from langchain_text_splitters import TokenTextSplitter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from together import AsyncTogether

from ypl.backend.config import settings
from ypl.backend.db import get_async_session
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import StopWatch
from ypl.db.embeddings import ChatMessageEmbedding

BGE_M3 = "BAAI/bge-m3"
BGE_LARGE_EN_V1_5 = "BAAI/bge-large-en-v1.5"
M2_BERT_80M_8K_RETRIEVAL = "togethercomputer/m2-bert-80M-8k-retrieval"
SUPPORTED_EMBEDDING_MODELS_CONTEXT_LENGTHS = {
    # High quality multiligual model, short context model. Output dimension: 1024.
    BGE_LARGE_EN_V1_5: 512,
    # High quality multilingual model, long context model. Output dimension: 768.
    M2_BERT_80M_8K_RETRIEVAL: 8192,
    # High quality multilingual model, long context model. Output dimension: 1024.
    BGE_M3: 8192,
}
PROVIDER = Literal["together", "internal"]
EMBEDDING_PROVIDERS: dict[str, PROVIDER] = {
    BGE_M3: "internal",
    BGE_LARGE_EN_V1_5: "together",
    M2_BERT_80M_8K_RETRIEVAL: "together",
}
DEFAULT_EMBEDDING_MODEL = BGE_M3
DEFAULT_TOKENIZATION_MODEL = "gpt-4"
DEFAULT_EMBEDDING_DIMENSION = 1536

# TODO(gilad): read from env, and split to staging and production.
INTERNAL_EMBEDDING_ENDPOINT = "http://embed.yupp.ai/embed"

# Multipliers to decrease the context length for input length validation exceptions.
CONTEXT_LENGTH_MULTIPLIERS = [0.95, 0.8, 0.6]


def _process_input(
    input: str,
    chunk_size: int,
    overlap_fraction: float = 0.05,
    tokenization_model: str = DEFAULT_TOKENIZATION_MODEL,
) -> list[str]:
    """Prepare the input for embedding by splitting it into chunks with overlap."""
    if overlap_fraction < 0.0 or overlap_fraction >= 1.0:
        raise ValueError(f"Invalid overlap fraction: {overlap_fraction}")
    chunk_overlap = int(chunk_size * overlap_fraction)
    return TokenTextSplitter(
        model_name=tokenization_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).split_text(input)


class InputLengthValidationException(Exception):
    pass


async def _embed_by_provider(
    provider: PROVIDER, inputs: list[str], embedding_model: str, max_embeddings: int | None = None
) -> list[list[list[float]]]:
    if provider == "together":
        if "TOGETHER_API_KEY" not in os.environ:
            raise ValueError("TOGETHER_API_KEY is not set")
    elif provider != "internal":
        raise ValueError(f"Unsupported provider: {provider}")

    return await _embed_with_retry(provider, inputs, embedding_model, max_embeddings)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    retry=retry_if_exception_type(
        InputLengthValidationException,
    ),
)
async def _embed_with_retry(
    provider: PROVIDER, inputs: list[str], embedding_model: str, max_embeddings: int | None = None
) -> list[list[list[float]]]:
    # The context length computed by the tokenization model is not always consistent with the context length
    # computed by the embedding model. When the embedding model errors due to too-long input, we retry with a
    # smaller context length.
    attempt_number = _embed_with_retry.statistics["attempt_number"]  # type: ignore
    context_length_reduction_multiplier = CONTEXT_LENGTH_MULTIPLIERS[attempt_number - 1]
    context_length = int(
        SUPPORTED_EMBEDDING_MODELS_CONTEXT_LENGTHS[embedding_model] * context_length_reduction_multiplier
    )

    stopwatch = StopWatch()
    processed_inputs = [_process_input(input, chunk_size=context_length) for input in inputs]
    stopwatch.record_split("process_input")

    if max_embeddings is not None:
        processed_inputs = [processed_input[:max_embeddings] for processed_input in processed_inputs]

    # processed_inputs is a list of lists of strings: one list per input string, as that string is chunked to sizes that
    # the embedding model can handle.
    # The embedding model takes a single flat list, so first flatten processed_inputs, then embed it, and finally
    # re-batch the embeddings into the same structure as processed_inputs.

    flat_processed_inputs = [item for sublist in processed_inputs for item in sublist]
    if provider == "together":
        embeddings = await _embed_together(flat_processed_inputs, embedding_model)
    elif provider == "internal":
        embeddings = await _embed_internal(flat_processed_inputs, embedding_model)
    stopwatch.end("embed")
    logging.info(
        json_dumps(
            {
                "message": f"Embedding using '{provider}' provider",
                "context_length": context_length,
                "num_inputs": len(processed_inputs),
                "processed_input_lengths_chars": [
                    [len(chunk) for chunk in processed_input] for processed_input in processed_inputs
                ],
                "timing": stopwatch.splits,
                "attempt_number": attempt_number,
            }
        )
    )
    stopwatch.end()

    # Group embeddings into sublists matching the input structure.
    lengths = [len(sublist) for sublist in processed_inputs]
    indices = [sum(lengths[:i]) for i in range(len(lengths) + 1)]
    batched_embeddings = [embeddings[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)]
    assert len(batched_embeddings) == len(processed_inputs)
    assert all(
        len(batched_embedding) == len(processed_input)
        for batched_embedding, processed_input in zip(batched_embeddings, processed_inputs, strict=True)
    )
    return batched_embeddings


async def _embed_together(inputs: list[str], embedding_model: str) -> list[list[float]]:
    try:
        response = await AsyncTogether().embeddings.create(input=inputs, model=embedding_model)
    except Exception as e:
        if "Input validation error: `inputs` tokens" in str(e):
            raise InputLengthValidationException(
                f"Input too long for model {embedding_model}; retrying with smaller context length"
            ) from e
        logging.error(f"Error embedding input: {e}")
        raise e
    return [x.embedding for x in response.data]


async def _embed_internal(inputs: list[str], embedding_model: str) -> list[list[float]]:
    headers = {"Content-Type": "application/json", "x-api-key": settings.embed_x_api_key}
    payload = {"texts": inputs}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(INTERNAL_EMBEDDING_ENDPOINT, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logging.error(f"Error from internal embedding service: {error_text}")
                    raise Exception(f"Internal embedding service returned status {response.status}: {error_text}")

                result = await response.json()
                return result["embeddings"]  # type: ignore
    except aiohttp.ClientError as e:
        logging.error(f"Error connecting to internal embedding service: {e}")
        raise Exception(f"Failed to connect to internal embedding service: {e}") from e
    except Exception as e:
        err = str(e).lower()
        if "out of memory" in err or "inputs exceed max length" in err:
            raise InputLengthValidationException(
                f"Input too long for model {embedding_model}; retrying with smaller context length"
            ) from e
        logging.error(f"Error embedding input with internal service: {e}")
        raise e


async def embed(
    inputs: list[str],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    max_embeddings: int | None = None,
    pad_to_length: int | None = None,
) -> list[list[list[float]]]:
    """
    Embed a list of strings.

    Args:
        inputs: The list of strings to embed.
        embedding_model: The embedding model to use.
        max_embeddings: The maximum number of embeddings to return.
        pad_to_length: The length to pad the embeddings to.

    Returns:
        A list of lists of embeddings. Each input string is split into chunks (up to `max_embeddings` per input),
        and each chunk is embedded:
        - The outer list is the inputs.
        - The middle list is the chunks of a single input.
        - The inner list is the embedding of a single chunk, as a list of floats.
    """
    provider = EMBEDDING_PROVIDERS.get(embedding_model)
    if not provider:
        raise ValueError(f"No provider for model: {embedding_model}")
    if embedding_model not in SUPPORTED_EMBEDDING_MODELS_CONTEXT_LENGTHS:
        raise ValueError(f"Unsupported model: {embedding_model}")

    embeddings = await _embed_by_provider(provider, inputs, embedding_model, max_embeddings)

    if pad_to_length:
        embeddings = [[x + [0.0] * (pad_to_length - len(x)) for x in embedding] for embedding in embeddings]
    return embeddings


async def embed_and_store_chat_message_embeddings(message_id: uuid.UUID, message_content: str) -> None:
    """Embed the message content and store the resulting embeddings."""
    embeddings = await embed(
        [message_content], embedding_model=DEFAULT_EMBEDDING_MODEL, pad_to_length=DEFAULT_EMBEDDING_DIMENSION
    )
    async with get_async_session() as session:
        for embedding in embeddings[0]:
            cme = ChatMessageEmbedding(
                message_id=message_id,
                embedding=embedding,
                embedding_model_name=DEFAULT_EMBEDDING_MODEL,
            )
            session.add(cme)
        await session.commit()
