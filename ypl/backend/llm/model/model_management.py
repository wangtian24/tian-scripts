# Standard library imports
import logging
import os
from collections.abc import Sequence

from sqlalchemy.exc import DatabaseError, OperationalError
from sqlmodel import desc, select, update
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from ypl.backend.db import get_async_session
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

DATABASE_ATTEMPTS = 3
DATABASE_WAIT_TIME_SECONDS = 0.1
DATABASE_RETRY_IF_EXCEPTION_TYPE = (OperationalError, DatabaseError)


@retry(
    stop=stop_after_attempt(DATABASE_ATTEMPTS),
    wait=wait_fixed(DATABASE_WAIT_TIME_SECONDS),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type(DATABASE_RETRY_IF_EXCEPTION_TYPE),
)
async def _get_active_models() -> Sequence[tuple[LanguageModel, str, str]]:
    async with get_async_session() as session:
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
        result = await session.exec(query)
        return result.all()


async def validate_active_onboarded_models() -> None:
    """
    Validate active models.

    This function should be run periodically to check and update the status of active models.
    It queries for active models, verifies them, and logs an error if any validation fails.
    """
    active_models = await _get_active_models()
    for model, provider_name, base_url in active_models:
        await verify_and_update_model_status(model, provider_name, base_url)


@retry(
    stop=stop_after_attempt(DATABASE_ATTEMPTS),
    wait=wait_fixed(DATABASE_WAIT_TIME_SECONDS),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type(DATABASE_RETRY_IF_EXCEPTION_TYPE),
)
async def _update_model_status(model: LanguageModel, status: LanguageModelStatusEnum) -> None:
    async with get_async_session() as session:
        await session.exec(
            update(LanguageModel)
            .values(status=status)
            .where(LanguageModel.language_model_id == model.language_model_id)  # type: ignore
        )
        await session.commit()


@retry(
    stop=stop_after_attempt(DATABASE_ATTEMPTS),
    wait=wait_fixed(DATABASE_WAIT_TIME_SECONDS),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type(DATABASE_RETRY_IF_EXCEPTION_TYPE),
)
async def _insert_language_model_response_status(
    model: LanguageModel, status_type: LanguageModelResponseStatusEnum
) -> None:
    async with get_async_session() as session:
        new_language_model_response_record = LanguageModelResponseStatus(
            language_model_id=model.language_model_id,
            status_type=status_type,
        )
        session.add(new_language_model_response_record)
        await session.commit()


async def verify_and_update_model_status(model: LanguageModel, provider_name: str, base_url: str) -> None:
    """
    Verify and update the status of a single active model.
    If the model is not running on 2 consecutive runs, set the model to INACTIVE.

    Args:
        model (LanguageModel): The model to verify and update.
        provider_name (str): The name of the provider.
        base_url (str): The base URL for the provider's API.
    """
    try:
        async with get_async_session() as session:
            last_status_query = (
                select(LanguageModelResponseStatus)
                .where(
                    LanguageModelResponseStatus.language_model_id == model.language_model_id,
                    LanguageModelResponseStatus.status_type.in_(INFERENCE_STATUS_TYPES),  # type: ignore
                )
                .order_by(desc(LanguageModelResponseStatus.created_at))
                .limit(1)
            )
            last_status_result = await session.exec(last_status_query)
            last_status = last_status_result.one_or_none()

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
            await _insert_language_model_response_status(model, current_status_type)

        if not is_inference_running:
            if last_status and last_status.status_type == LanguageModelResponseStatusEnum.INFERENCE_FAILED:
                await _update_model_status(model, LanguageModelStatusEnum.INACTIVE)
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

        await _insert_language_model_response_status(model, LanguageModelResponseStatusEnum.INFERENCE_FAILED)

        slack_message = (
            f"Environment {os.environ.get('ENVIRONMENT')} - Model {model.name} from "
            f"{provider_name} inference validation failed: {str(e)}"
        )
        await post_to_slack(f":x: {slack_message}")
