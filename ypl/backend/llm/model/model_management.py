# Standard library imports
import logging
import os
from uuid import UUID

# Third-party imports
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

# Local imports
from ypl.backend.db import get_async_engine
from ypl.backend.llm.model.model_onboarding import verify_inference_running
from ypl.backend.llm.utils import post_to_slack
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, Provider


async def validate_active_onboarded_models() -> None:
    """
    Validate active models.

    This function should be run periodically to check and update the status of active models.
    It queries for active models, verifies them, and logs an error if any validation fails.
    """
    async with AsyncSession(get_async_engine()) as session:
        query = (
            select(LanguageModel, Provider.name.label("provider_name"), Provider.base_api_url.label("base_url"))  # type: ignore
            .join(Provider, LanguageModel.provider_id == Provider.provider_id)  # type: ignore
            .where(
                LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
                LanguageModel.provider_id.is_not(None),  # type: ignore
                LanguageModel.deleted_at.is_(None),  # type: ignore
            )
        )
        active_models = await session.execute(query)

        for model, provider_name, base_url in active_models:
            await verify_and_update_model_status(session, model, provider_name, base_url)

        await session.commit()


async def validate_specific_active_model(model_id: UUID) -> None:
    """
    Validate a specific active model.

    This function should be called to check and update the status of a specific active model.
    It queries for the model, verifies it, and logs an error if validation fails.
    """
    async with AsyncSession(get_async_engine()) as session:
        query = (
            select(LanguageModel, Provider.name.label("provider_name"), Provider.base_api_url.label("base_url"))  # type: ignore
            .join(Provider, LanguageModel.provider_id == Provider.provider_id)  # type: ignore
            .where(
                LanguageModel.language_model_id == model_id,
                LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
                LanguageModel.provider_id.is_not(None),  # type: ignore
                LanguageModel.deleted_at.is_(None),  # type: ignore
            )
        )
        result = await session.execute(query)

        if result:
            model, provider_name, base_url = result
            await verify_and_update_model_status(session, model, provider_name, base_url)
            await session.commit()
        else:
            logging.warning(f"No active model found with id {model_id}")


async def verify_and_update_model_status(
    session: AsyncSession, model: LanguageModel, provider_name: str, base_url: str
) -> None:
    """
    Verify and update the status of a single active model.

    Args:
        session (Session): The database session.
        model (LanguageModel): The model to verify and update.
        provider_name (str): The name of the provider.
        base_url (str): The base URL for the provider's API.
    """
    try:
        is_inference_running = verify_inference_running(model, provider_name, base_url)

        if not is_inference_running:
            # TODO: Uncomment once we have a way to track number of failures
            # and set the model to INACTIVE after a certain number of failures
            # model.status = LanguageModelStatusEnum.INACTIVE
            # model.modified_at = datetime.utcnow()
            error_message = (
                f"Environment {os.environ.get('ENVIRONMENT')} - Model {model.name} ({model.internal_name}) "
                "is not running at the inference endpoint. Please investigate."
            )
            logging.error(error_message)
            await post_to_slack(f":warning: {error_message}")
    except Exception:
        error_message = (
            f"Environment {os.environ.get('ENVIRONMENT')} - Model {model.name} ({model.internal_name}) "
            "inference validation failed: {str(e)}"
        )
        logging.error(error_message)
        await post_to_slack(f":x: {error_message}")
        # TODO: Implement additional alerting if needed
