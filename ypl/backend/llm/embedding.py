import logging
import os

from langchain_text_splitters import TokenTextSplitter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from together import AsyncTogether

from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import StopWatch

BGE_LARGE_EN_V1_5 = "BAAI/bge-large-en-v1.5"
M2_BERT_80M_8K_RETRIEVAL = "togethercomputer/m2-bert-80M-8k-retrieval"
SUPPORTED_EMBEDDING_MODELS_CONTEXT_LENGTHS = {
    # High quality multiligual model, short context model. Output dimension: 1024.
    BGE_LARGE_EN_V1_5: 512,
    # High quality multilingual model, long context model. Output dimension: 768.
    M2_BERT_80M_8K_RETRIEVAL: 8192,
}
DEFAULT_TOGETHER_MODEL = BGE_LARGE_EN_V1_5
DEFAULT_TOKENIZATION_MODEL = "gpt-4"

# Multipliers to decrease the context length for input length validation exceptions.
CONTEXT_LENGTH_MULTIPLIERS = [0.9, 0.75, 0.5]


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(0.1),
    retry=retry_if_exception_type(
        InputLengthValidationException,
    ),
)
async def _embed_together(input: str, embedding_model: str, max_embeddings: int | None = None) -> list[list[float]]:
    if "TOGETHER_API_KEY" not in os.environ:
        raise ValueError("TOGETHER_API_KEY is not set")
    if embedding_model not in SUPPORTED_EMBEDDING_MODELS_CONTEXT_LENGTHS:
        raise ValueError(f"Unsupported model: {embedding_model}")

    # The context length computed by the tokenization model is not always consistent with the context length
    # computed by the embedding model. When the embedding model errors due to too-long input, we retry with a
    # smaller context length.
    attempt_number = _embed_together.statistics["attempt_number"]  # type: ignore
    context_length_reduction_multiplier = CONTEXT_LENGTH_MULTIPLIERS[attempt_number - 1]
    context_length = int(
        SUPPORTED_EMBEDDING_MODELS_CONTEXT_LENGTHS[embedding_model] * context_length_reduction_multiplier
    )

    stopwatch = StopWatch()
    processed_input = _process_input(input, chunk_size=context_length)
    stopwatch.record_split("process_input")

    if max_embeddings is not None:
        processed_input = processed_input[:max_embeddings]

    try:
        response = await AsyncTogether().embeddings.create(input=processed_input, model=embedding_model)
    except Exception as e:
        if "Input validation error: `inputs` tokens" in str(e):
            raise InputLengthValidationException(
                f"Input too long for model {embedding_model}; retrying with smaller context length"
            ) from e
        logging.error(f"Error embedding input: {e}")
        raise e
    stopwatch.end("embed")
    logging.info(
        json_dumps(
            {
                "message": "Embedding using Together",
                "input_length_chars": len(input),
                "context_length": context_length,
                "processed_input_lengths_chars": [len(chunk) for chunk in processed_input],
                "timing": stopwatch.splits,
                "attempt_number": attempt_number,
            }
        )
    )
    return [x.embedding for x in response.data]


async def embed(
    input: str,
    embedding_model: str = DEFAULT_TOGETHER_MODEL,
    max_embeddings: int | None = None,
    pad_to_length: int | None = None,
) -> list[list[float]]:
    if embedding_model in (BGE_LARGE_EN_V1_5, M2_BERT_80M_8K_RETRIEVAL):
        embeddings = await _embed_together(input, embedding_model, max_embeddings)
    else:
        raise ValueError(f"Unsupported model: {embedding_model}")
    if pad_to_length:
        embeddings = [x + [0.0] * (pad_to_length - len(x)) for x in embeddings]
    return embeddings
