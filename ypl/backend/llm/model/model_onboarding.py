# Standard library imports
import logging
import os
from datetime import datetime

# Third-party imports
from huggingface_hub import HfApi, ModelInfo
from huggingface_hub.utils import HfHubHTTPError
from openai import OpenAI
from sqlmodel import Session, select

# Local imports
from ypl.backend.db import get_engine
from ypl.backend.llm.constants import PROVIDER_KEY_MAPPING
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, Provider

# Constants
MMLU_PRO_THRESHOLD = 50
DOWNLOADS_THRESHOLD = 1000
LIKES_THRESHOLD = 100


def verify_onboard_submitted_models() -> None:
    """
    Verify and onboard submitted models.

    This function should be run periodically to check and update the status of submitted models.
    It queries for submitted models, verifies them, and updates their status accordingly.
    """
    with Session(get_engine()) as session:
        query = (
            select(LanguageModel, Provider.name.label("provider_name"), Provider.base_api_url.label("base_url"))  # type: ignore
            .join(Provider, LanguageModel.provider_id == Provider.provider_id)  # type: ignore
            .where(
                LanguageModel.status == LanguageModelStatusEnum.SUBMITTED,
                LanguageModel.deleted_at.is_(None),  # type: ignore
            )
        )
        submitted_models = session.exec(query).all()

        for model, provider_name, base_url in submitted_models:
            try:
                is_hf_verified = verify_hf_model(model)
                is_inference_running = verify_inference_running(model, provider_name, base_url)

                if is_hf_verified or is_inference_running:
                    model.status = LanguageModelStatusEnum.VERIFIED_PENDING_ACTIVATION
                    model.modified_at = datetime.utcnow()
                    logging.info(
                        f"Model {model.name} ({model.internal_name}) validated successfully "
                        f"and set to VERIFIED_PENDING_ACTIVATION."
                    )
                else:
                    model.status = LanguageModelStatusEnum.REJECTED
                    model.modified_at = datetime.utcnow()
                    logging.info(
                        f"Model {model.name} ({model.internal_name}) not validated. " f"Setting status to REJECTED."
                    )
            except Exception as e:
                # TODO: Implement alerting
                logging.error(f"Model {model.name} ({model.internal_name}) validation failed: {str(e)}")

        session.commit()


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

        logging.info(
            f"Model {model.internal_name} - Tags: "
            f"text-generation: {has_text_generation}, "
            f"conversational: {has_conversational}, "
            f"endpoints_compatible: {has_endpoints_compatible}"
        )

        downloads = model_info.downloads or 0
        likes = model_info.likes or 0
        logging.info(
            f"Model {model.internal_name} - Downloads: {downloads}, "
            f"Likes: {likes}, MMLU-PRO score: {mmlu_pro_score}"
        )

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
        logging.error(f"Error accessing Hugging Face API for model {model.internal_name}: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error verifying model {model.internal_name} on Hugging Face: {str(e)}")
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
    Verify if the model is running on PyTorch Serve.
    """
    try:
        # TODO: Update this code to ensure non OpenAI providers are supported
        api_key = get_provider_api_key(provider_name)
        client = OpenAI(api_key=api_key, base_url=base_url)

        completion = client.chat.completions.create(
            model=model.internal_name, messages=[{"role": "user", "content": "What is the capital of Odisha?"}]
        )

        if completion.choices[0].message.content and len(completion.choices[0].message.content) > 0:
            logging.info(f"Model {model.internal_name} is running on provider's endpoint.")
            return True
        else:
            return False
    except Exception as e:
        logging.error(f"Unexpected error verifying inference running for model {model.internal_name}: {str(e)}")
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
    cleaned_provider_name = "".join(provider_name.split()).lower()
    env_var_name = PROVIDER_KEY_MAPPING.get(cleaned_provider_name)

    if not env_var_name:
        raise ValueError(f"Unknown provider name: {provider_name}")

    api_key = os.environ.get(env_var_name)
    if not api_key:
        raise ValueError(f"API key not found for provider: {provider_name}")

    return api_key


if __name__ == "__main__":
    verify_onboard_submitted_models()
