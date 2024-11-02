# Standard library imports
import logging
import os
import time
from datetime import UTC, datetime
from functools import cache
from typing import Any
from uuid import UUID

# Third-party imports
import anthropic
import google.generativeai as genai
from huggingface_hub import HfApi, InferenceClient, ModelInfo
from huggingface_hub.utils import HfHubHTTPError
from openai import OpenAI
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tenacity.asyncio import AsyncRetrying

# Local imports
from ypl.backend.db import get_async_engine
from ypl.backend.llm.chat import standardize_provider_name
from ypl.backend.llm.constants import PROVIDER_KEY_MAPPING
from ypl.backend.llm.utils import post_to_slack, post_to_x
from ypl.backend.utils.json import json_dumps
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, Provider

# Constants
MMLU_PRO_THRESHOLD = 50
DOWNLOADS_THRESHOLD = 1000
LIKES_THRESHOLD = 100
REJECT_AFTER_DAYS = 3
MAX_RETRIES = 3
WAIT_TIME = 5
INFERENCE_TIMEOUT = 60

# Replace the existing client caches with a single cache
provider_clients: dict[str, Any] = {}


@cache
def get_provider_client(
    provider_name: str, api_key: str, model_name: str | None = None, base_url: str | None = None
) -> Any:
    """Get or create a client for the specified provider."""
    if provider_name not in provider_clients:
        if provider_name == "google":
            genai.configure(api_key=api_key)
            provider_clients[provider_name] = genai.GenerativeModel(model_name)
        elif provider_name == "huggingface":
            provider_clients[provider_name] = InferenceClient(token=api_key)
        elif provider_name == "anthropic":
            provider_clients[provider_name] = anthropic.Anthropic(api_key=api_key)
        else:  # assumes OpenAI and compatible providers
            provider_clients[provider_name] = OpenAI(api_key=api_key, base_url=base_url)
    return provider_clients[provider_name]


async def async_retry_decorator() -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_fixed(WAIT_TIME),
        after=after_log(logging.getLogger(), logging.WARNING),
        retry=retry_if_exception_type((OperationalError, DatabaseError)),
    )


async def verify_onboard_submitted_models() -> None:
    """
    Verify and onboard submitted models.

    This function should be run periodically to check and update the status of submitted models.
    It queries for submitted models, verifies them, and updates their status accordingly.
    """
    async for attempt in await async_retry_decorator():
        with attempt:
            async with AsyncSession(get_async_engine()) as session:
                query = (
                    select(LanguageModel, Provider.name.label("provider_name"), Provider.base_api_url.label("base_url"))  # type: ignore
                    .join(Provider, LanguageModel.provider_id == Provider.provider_id)  # type: ignore
                    .where(
                        LanguageModel.status == LanguageModelStatusEnum.SUBMITTED,
                        LanguageModel.deleted_at.is_(None),  # type: ignore
                    )
                )
                submitted_models = await session.execute(query)

                for model, provider_name, base_url in submitted_models:
                    await verify_and_update_model_status(session, model, provider_name, base_url)

                await session.commit()


async def verify_onboard_specific_model(model_id: UUID) -> None:
    """
    Verify and onboard a specific model.

    This function should be called to check and update the status of a specific model.
    It queries for the model, verifies it, and updates its status accordingly.
    """
    async for attempt in await async_retry_decorator():
        with attempt:
            async with AsyncSession(get_async_engine()) as session:
                query = (
                    select(LanguageModel, Provider.name.label("provider_name"), Provider.base_api_url.label("base_url"))  # type: ignore
                    .join(Provider, LanguageModel.provider_id == Provider.provider_id)  # type: ignore
                    .where(
                        LanguageModel.language_model_id == model_id,
                        LanguageModel.status == LanguageModelStatusEnum.SUBMITTED,
                        LanguageModel.deleted_at.is_(None),  # type: ignore
                    )
                )
                result = await session.execute(query)
                row = result.first()

                if row:
                    model, provider_name, base_url = row
                    await verify_and_update_model_status(session, model, provider_name, base_url)
                    await session.commit()
                else:
                    log_dict = {"message": f"No submitted model found with id {model_id}", "model_id": model_id}
                    logging.warning(json_dumps(log_dict))


async def verify_and_update_model_status(
    session: AsyncSession, model: LanguageModel, provider_name: str, base_url: str
) -> None:
    """
    Verify and update the status of a single model.

    Args:
        session (AsyncSession): The database session.
        model (LanguageModel): The model to verify and update.
        provider_name (str): The name of the provider.
        base_url (str): The base URL for the provider's API.
    """
    async for attempt in await async_retry_decorator():
        with attempt:
            try:
                is_inference_running = verify_inference_running(model, provider_name, base_url)

                if is_inference_running:
                    model.status = LanguageModelStatusEnum.ACTIVE
                    model.modified_at = datetime.utcnow()
                    log_dict = {
                        "message": "Model validated successfully and set to ACTIVE",
                        "model_name": model.name,
                    }
                    logging.info(json_dumps(log_dict))
                    log_message = (
                        f"Environment {os.environ.get('ENVIRONMENT')} - Model {model.name} "
                        "validated successfully and set to ACTIVE."
                    )
                    await post_to_slack(log_message)
                    await post_to_x(log_message)
                else:
                    is_hf_verified = verify_hf_model(model)
                    if is_hf_verified:
                        model.status = LanguageModelStatusEnum.VERIFIED_PENDING_ACTIVATION
                        model.modified_at = datetime.utcnow()
                        log_dict = {
                            "message": "Model validated successfully and set to VERIFIED_PENDING_ACTIVATION",
                            "model_name": model.name,
                        }
                        logging.info(json_dumps(log_dict))
                        log_message = (
                            f"Environment {os.environ.get('ENVIRONMENT')} - Model {model.name} "
                            "validated successfully and set to VERIFIED_PENDING_ACTIVATION."
                        )
                        await post_to_slack(log_message)
                        await post_to_x(log_message)
                    else:
                        # reject if the model was submitted more than 3 days ago
                        if (
                            model.created_at
                            and (datetime.now(UTC) - model.created_at.replace(tzinfo=UTC)).days > REJECT_AFTER_DAYS
                        ):
                            model.status = LanguageModelStatusEnum.REJECTED
                            model.modified_at = datetime.utcnow()
                            log_dict = {
                                "message": "Model not validated after 3 days. Setting status to REJECTED",
                                "model_name": model.name,
                            }
                            logging.info(json_dumps(log_dict))
                            log_message = (
                                f"Environment {os.environ.get('ENVIRONMENT')} - "
                                f"Model {model.name}"
                                "not validated after 3 days. Setting status to REJECTED."
                            )
                            await post_to_slack(log_message)

                await session.commit()

            except Exception as e:
                log_dict = {
                    "message": "Model validation failed",
                    "model_name": model.name,
                    "error": str(e),
                }
                logging.exception(json_dumps(log_dict))
                await post_to_slack(
                    f"Environment {os.environ.get('ENVIRONMENT')} - Model {model.name} " f"validation failed: {str(e)}"
                )
                # Re-raise the exception to trigger a retry
                raise


def verify_hf_model(model: LanguageModel) -> bool:
    """
    Verify a model on Hugging Face.

    Args:
        model (LanguageModel): The model to verify.

    Returns:
        bool: True if the model is verified, False otherwise.

    Raises:
        HfHubHTTPError: If there's an error accessing the Hugging Face API.
    """
    try:
        api = HfApi()
        model_info = api.model_info(model.internal_name)
        mmlu_pro_score = get_mmlu_pro_score(model_info)

        # Check for specific tags
        tags = model_info.tags or []
        has_text_generation = "text-generation" in tags
        has_conversational = "conversational" in tags
        has_endpoints_compatible = "endpoints_compatible" in tags

        downloads = model_info.downloads or 0
        likes = model_info.likes or 0

        log_dict = {
            "message": "Model characteristics",
            "model_name": model.name,
            "tags": tags,
            "has_text_generation": has_text_generation,
            "has_conversational": has_conversational,
            "has_endpoints_compatible": has_endpoints_compatible,
            "downloads": downloads,
            "likes": likes,
            "mmlu_pro_score": mmlu_pro_score,
        }
        logging.info(json_dumps(log_dict))
        # Return true if the model has more than 1000 downloads and 100 likes
        # and has the required tags for text-generation, conversational and endpoints_compatible
        is_verified = (
            downloads > DOWNLOADS_THRESHOLD
            and likes > LIKES_THRESHOLD
            and all([has_text_generation, has_conversational, has_endpoints_compatible])
        )

        # Return true if the model has more than 50 MMLU-PRO score
        # and has the required tags for text-generation, conversational and endpoints_compatible
        is_verified = is_verified or (
            mmlu_pro_score > MMLU_PRO_THRESHOLD
            and all([has_text_generation, has_conversational, has_endpoints_compatible])
        )

        return is_verified
    except HfHubHTTPError as e:
        log_dict = {
            "message": "Error accessing Hugging Face API for model",
            "model_name": model.name,
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        return False
    except Exception as e:
        log_dict = {
            "message": "Unexpected error verifying model on Hugging Face",
            "model_name": model.name,
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        return False


def get_mmlu_pro_score(model_info: ModelInfo) -> float:
    """
    Extract the MMLU-PRO score from the model info.

    Args:
        model_info (ModelInfo): The model information from Hugging Face.

    Returns:
        float: The MMLU-PRO score, or 0 if not found.
    """
    if isinstance(model_info.model_index, list) and model_info.model_index:
        for result in model_info.model_index[0].get("results", []):
            dataset = result.get("dataset", {})
            if dataset.get("name", "").startswith("MMLU-PRO"):
                for metric in result.get("metrics", []):
                    if metric.get("type") == "acc":
                        return float(metric.get("value", 0))
    return 0


def verify_inference_running(model: LanguageModel, provider_name: str, base_url: str) -> bool:
    """
    Verify if the model is running on the provider's endpoint.
    """
    try:
        is_inference_running = False
        api_key = get_provider_api_key(provider_name)
        cleaned_provider_name = standardize_provider_name(provider_name)

        client = get_provider_client(cleaned_provider_name, api_key, model.internal_name, base_url)

        start_time = time.time()
        if cleaned_provider_name == "anthropic":
            is_inference_running = anthropic_api_call(client, model.internal_name)
        elif cleaned_provider_name == "huggingface":
            is_inference_running = huggingface_api_call(client, model.internal_name)
        elif cleaned_provider_name == "google":
            content = google_ai_api_call(client, model.internal_name)
            is_inference_running = bool(content.text and len(content.text) > 0)
        else:  # OpenAI and compatible providers
            completion = openai_api_call(client, model.internal_name)
            is_inference_running = bool(
                completion.choices[0].message.content and len(completion.choices[0].message.content) > 0
            )
        end_time = time.time()
        latency = round(end_time - start_time, 3)

        if is_inference_running:
            log_dict = {
                "message": f"Inference running latency for {model.name} - latency: {latency} seconds",
                "model_name": model.name,
                "cleaned_provider_name": cleaned_provider_name,
                "latency": latency,
            }
            logging.info(json_dumps(log_dict))

        return is_inference_running
    except Exception as e:
        log_dict = {
            "message": "Unexpected error verifying inference running for model",
            "model_name": model.name,
            "provider_name": provider_name,
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        return False


def get_provider_api_key(provider_name: str) -> str:
    """
    Get the API key for a specific provider.

    Args:
        provider_name (str): The name of the provider.

    Returns:
        str: The API key for the specified provider.

    Raises:
        ValueError: If the API key for the provider is not found.
    """
    # Remove any blank characters within the provider_name as DB has blank spaces
    cleaned_provider_name = standardize_provider_name(provider_name)
    env_var_name = PROVIDER_KEY_MAPPING.get(cleaned_provider_name)
    log_dict = {
        "message": "API key environment variable name for provider",
        "provider_name": provider_name,
        "env_var_name": env_var_name,
    }
    logging.debug(json_dumps(log_dict))

    if not env_var_name:
        log_dict = {
            "message": "Unknown provider name",
            "provider_name": provider_name,
        }
        logging.error(json_dumps(log_dict))
        raise ValueError(f"Unknown provider name: {provider_name}")

    api_key = os.environ.get(env_var_name)
    if not api_key:
        log_dict = {
            "message": "API key not found for provider",
            "provider_name": provider_name,
        }
        logging.error(json_dumps(log_dict))
        raise ValueError(f"API key not found for provider: {provider_name}")

    return api_key


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_fixed(WAIT_TIME),
    retry=retry_if_exception_type((Exception, TimeoutError)),
)
def openai_api_call(client: Any, model_name: str) -> Any:
    return client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "What is the capital of Odisha?"}],
        timeout=INFERENCE_TIMEOUT,
    )


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_fixed(WAIT_TIME),
    retry=retry_if_exception_type((Exception, TimeoutError)),
)
def google_ai_api_call(client: Any, model_name: str) -> Any:
    return client.generate_content("What is the capital of Odisha?")


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_fixed(WAIT_TIME),
    retry=retry_if_exception_type((Exception, TimeoutError)),
)
def huggingface_api_call(client: Any, model_name: str) -> bool:
    messages = [
        {"role": "user", "content": "Tell me a story"},
    ]
    completion = client.chat.completions.create(model=model_name, messages=messages, stream=True)

    for chunk in completion:
        if chunk.choices[0].delta.content and len(chunk.choices[0].delta.content) > 0:
            return True
    return False


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_fixed(WAIT_TIME),
    retry=retry_if_exception_type((Exception, TimeoutError)),
)
def anthropic_api_call(client: Any, model_name: str) -> bool:
    message = client.messages.create(
        model=model_name,
        max_tokens=1024,
        messages=[{"role": "user", "content": "What is the capital of Odisha?"}],
        timeout=INFERENCE_TIMEOUT,
    )
    return bool(message.content and len(message.content) > 0)
