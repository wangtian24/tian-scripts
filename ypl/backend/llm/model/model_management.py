# Standard library imports
import logging
from collections.abc import Sequence

from sqlalchemy.exc import DatabaseError, OperationalError
from sqlmodel import desc, select, update
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from ypl.backend.db import get_async_session
from ypl.backend.llm.model.management_common import ModelManagementStatus, _verify_inference_running, log_and_post
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
async def _get_active_models() -> Sequence[tuple[LanguageModel, str]]:
    async with get_async_session() as session:
        query = (
            select(LanguageModel, Provider.name.label("provider_name"))  # type: ignore
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


async def do_validate_active_models() -> None:
    """
    Validate active models.

    This function should be run periodically to check and update the status of active models.
    It queries for active models, verifies them, and logs an error if any validation fails.
    """
    active_models = await _get_active_models()
    print(f"Found {len(active_models)} active models: {', '.join([model.name for model, _ in active_models])}\n\n")

    for model, provider_name in active_models:
        await _validate_one_active_model(model, provider_name)


@retry(
    stop=stop_after_attempt(DATABASE_ATTEMPTS),
    wait=wait_fixed(DATABASE_WAIT_TIME_SECONDS),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type(DATABASE_RETRY_IF_EXCEPTION_TYPE),
)
async def _update_model_status(model: LanguageModel, status: LanguageModelStatusEnum) -> None:
    async with get_async_session() as session:
        print(f">> [DB] updating model {model.name} status to {status}")
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
        print(f">> [DB] updating model {model.name} response status to {status_type}")
        new_language_model_response_record = LanguageModelResponseStatus(
            language_model_id=model.language_model_id,
            status_type=status_type,
        )
        session.add(new_language_model_response_record)
        await session.commit()


async def _validate_one_active_model(model: LanguageModel, provider_name: str) -> None:
    """
    Verify and update the status of a single active model.
    If the model is not running on 2 consecutive runs, set the model to INACTIVE.

    Args:
        model (LanguageModel): The model to verify and update.
        provider_name (str): The name of the provider.
        base_url (str): The base URL for the provider's API.
    """
    try:
        model_name = model.name
        print(f"\nModel {model.name}: starting validation")
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

        is_inference_running, has_billing_error, excerpt = await _verify_inference_running(model)

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
            # The model is not running!
            print(f"Model {model_name}: done validation: Failed")
            if last_status and last_status.status_type == LanguageModelResponseStatusEnum.INFERENCE_FAILED:
                # deactivate if it also failed last time
                await _update_model_status(model, LanguageModelStatusEnum.INACTIVE)
                await log_and_post(ModelManagementStatus.VALIDATION_DEACTIVATED, model_name)
            else:
                await log_and_post(ModelManagementStatus.VALIDATION_NOTIFY_NOT_RUNNING, model_name)

            if has_billing_error:
                await log_and_post(ModelManagementStatus.VALIDATION_BILLING_ERROR, model_name, excerpt)

        else:
            # The model is running now!
            print(f"Model {model_name}: done validation: Success")
            if last_status and last_status.status_type == LanguageModelResponseStatusEnum.INFERENCE_FAILED:
                await log_and_post(ModelManagementStatus.VALIDATION_NOTIFY_RECOVERED, model_name)

    except Exception as e:
        print(f"Model {model_name}: error during validation: {e}")
        await _insert_language_model_response_status(model, LanguageModelResponseStatusEnum.INFERENCE_FAILED)
        await log_and_post(ModelManagementStatus.ERROR, model_name, str(e))
