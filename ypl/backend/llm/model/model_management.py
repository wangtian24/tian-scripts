# Standard library imports
import logging
import os
from uuid import UUID

# Third-party imports
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import desc, select
from tenacity import after_log, retry_if_exception_type, stop_after_attempt, wait_fixed
from tenacity.asyncio import AsyncRetrying

# Local imports
from ypl.backend.db import get_async_engine
from ypl.backend.llm.model.model_onboarding import verify_inference_running
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps
from ypl.db.language_models import (
    LanguageModel,
    LanguageModelResponseStatus,
    LanguageModelResponseStatusEnum,
    LanguageModelStatusEnum,
    Provider,
)

INFERENCE_STATUS_TYPES: list[LanguageModelResponseStatusEnum] = [
    LanguageModelResponseStatusEnum.INFERENCE_SUCCEEDED,
    LanguageModelResponseStatusEnum.INFERENCE_FAILED,
]


async def async_retry_decorator() -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.1),
        after=after_log(logging.getLogger(), logging.WARNING),
        retry=retry_if_exception_type((OperationalError, DatabaseError)),
    )


async def validate_active_onboarded_models() -> None:
    """
    Validate active models.

    This function should be run periodically to check and update the status of active models.
    It queries for active models, verifies them, and logs an error if any validation fails.
    """
    async for attempt in await async_retry_decorator():
        with attempt:
            async with AsyncSession(get_async_engine()) as session:
                query = (
                    select(LanguageModel, Provider.name.label("provider_name"), Provider.base_api_url.label("base_url"))  # type: ignore
                    .join(Provider, LanguageModel.provider_id == Provider.provider_id)  # type: ignore
                    .where(
                        LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
                        LanguageModel.deleted_at.is_(None),  # type: ignore
                        Provider.is_active.is_(True),  # type: ignore
                        Provider.deleted_at.is_(None),  # type: ignore
                    )
                )
                active_models = await session.execute(query)
                result = await session.execute(query)
                active_models = result.all()
                active_models_count = len(active_models)

                log_dict = {"message": f"Active models count: {active_models_count}"}
                logging.info(json_dumps(log_dict))

                for model, provider_name, base_url in active_models:
                    await verify_and_update_model_status(session, model, provider_name, base_url)

                await session.commit()


async def validate_specific_active_model(model_id: UUID) -> None:
    """
    Validate a specific active model.

    This function should be called to check and update the status of a specific active model.
    It queries for the model, verifies it, and logs an error if validation fails.
    """
    async for attempt in await async_retry_decorator():
        with attempt:
            async with AsyncSession(get_async_engine()) as session:
                query = (
                    select(LanguageModel, Provider.name.label("provider_name"), Provider.base_api_url.label("base_url"))  # type: ignore
                    .join(Provider, LanguageModel.provider_id == Provider.provider_id)  # type: ignore
                    .where(
                        LanguageModel.language_model_id == model_id,
                        LanguageModel.status == LanguageModelStatusEnum.ACTIVE,
                        LanguageModel.deleted_at.is_(None),  # type: ignore
                        Provider.is_active.is_(True),  # type: ignore
                        Provider.deleted_at.is_(None),  # type: ignore
                    )
                )
                result = await session.execute(query)

                if result:
                    model, provider_name, base_url = result.first()
                    await verify_and_update_model_status(session, model, provider_name, base_url)
                    await session.commit()
                else:
                    log_dict = {"message": f"No active model found with ID: {model_id}"}
                    logging.warning(json_dumps(log_dict))


async def verify_and_update_model_status(
    session: AsyncSession, model: LanguageModel, provider_name: str, base_url: str
) -> None:
    """
    Verify and update the status of a single active model.
    If the model is not running on 2 consecutive runs, set the model to INACTIVE.

    Args:
        session (Session): The database session.
        model (LanguageModel): The model to verify and update.
        provider_name (str): The name of the provider.
        base_url (str): The base URL for the provider's API.
    """
    try:
        last_status_query = (
            select(LanguageModelResponseStatus)
            .where(
                LanguageModelResponseStatus.language_model_id == model.language_model_id,
                LanguageModelResponseStatus.status_type.in_(INFERENCE_STATUS_TYPES),  # type: ignore
            )
            .order_by(desc(LanguageModelResponseStatus.created_at))
            .limit(1)
        )
        last_status_result = await session.execute(last_status_query)
        last_status = last_status_result.scalar_one_or_none()

        is_inference_running, has_billing_error = verify_inference_running(model, provider_name, base_url)

        current_status_type = (
            LanguageModelResponseStatusEnum.INFERENCE_SUCCEEDED
            if is_inference_running
            else LanguageModelResponseStatusEnum.INFERENCE_FAILED
        )

        should_create_language_model_response_record = (
            last_status is None
            or last_status.status_type != current_status_type
            or current_status_type == LanguageModelResponseStatusEnum.INFERENCE_FAILED
        )

        if should_create_language_model_response_record:
            new_language_model_response_record = LanguageModelResponseStatus(
                language_model_id=model.language_model_id,
                status_type=current_status_type,
            )
            session.add(new_language_model_response_record)

        if not is_inference_running:
            if last_status and last_status.status_type == LanguageModelResponseStatusEnum.INFERENCE_FAILED:
                model.status = LanguageModelStatusEnum.INACTIVE
                await session.merge(model)

                log_dict = {
                    "message": "Model has been set to inactive due to consecutive inference failures",
                    "model_name": model.name,
                    "provider_name": provider_name,
                    "base_url": base_url,
                }
                logging.error(json_dumps(log_dict))

                slack_message = (
                    f"Environment {os.environ.get('ENVIRONMENT')} - Model {model.name} "
                    f"has been set to INACTIVE due to consecutive inference failures on {provider_name}."
                )
            else:
                log_dict = {
                    "message": "Model is not running at the inference endpoint",
                    "model_name": model.name,
                    "provider_name": provider_name,
                    "base_url": base_url,
                }
                logging.error(json_dumps(log_dict))

                slack_message = (
                    f"Environment {os.environ.get('ENVIRONMENT')} - Model {model.name} "
                    f"is not running at the inference endpoint on {provider_name}. Please investigate."
                )

            if has_billing_error:
                slack_message += " (Potential billing error detected)"

            await post_to_slack(f":warning: {slack_message}")
        elif last_status and last_status.status_type == LanguageModelResponseStatusEnum.INFERENCE_FAILED:
            slack_message = (
                f"Environment {os.environ.get('ENVIRONMENT')} - Model {model.name} "
                f"has recovered and is now running successfully on {provider_name}."
            )
            await post_to_slack(f":white_check_mark: {slack_message}")

    except Exception as e:
        log_dict = {
            "message": f"Inference validation failed for model {model.name}",
            "model_name": model.name,
            "model_internal_name": model.internal_name,
            "provider_name": provider_name,
            "base_url": base_url,
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))

        error_status = LanguageModelResponseStatus(
            language_model_id=model.language_model_id, status_type=LanguageModelResponseStatusEnum.INFERENCE_FAILED
        )
        session.add(error_status)

        slack_message = (
            f"Environment {os.environ.get('ENVIRONMENT')} - Model {model.name} from "
            f"{provider_name} inference validation failed: {str(e)}"
        )
        await post_to_slack(f":x: {slack_message}")
