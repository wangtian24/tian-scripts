# Standard library imports
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

# Third-party imports
import httpx
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)
from tenacity.asyncio import AsyncRetrying

# Local imports
from ypl.backend.config import settings
from ypl.backend.db import get_async_engine
from ypl.backend.llm.model.management_common import (
    REJECT_AFTER_DAYS,
    ModelAlertLevel,
    ModelManagementStatus,
    log_and_post,
    verify_inference_running,
)
from ypl.backend.utils.json import json_dumps
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, Provider

# Constants
MMLU_PRO_THRESHOLD = 50
DOWNLOADS_THRESHOLD = 1000
LIKES_THRESHOLD = 100
MAX_RETRIES = 3
WAIT_TIME = 5
# Replace the existing client caches with a single cache
provider_clients: dict[str, Any] = {}


async def _async_retry_decorator() -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_fixed(WAIT_TIME),
        after=after_log(logging.getLogger(), logging.WARNING),
        retry=retry_if_exception_type((OperationalError, DatabaseError)),
    )


async def do_verify_submitted_models() -> None:
    """
    Verify and onboard submitted models.

    This function should be run periodically to check and update the status of submitted models.
    It queries for submitted models, verifies them, and updates their status accordingly.
    """
    async for attempt in await _async_retry_decorator():
        with attempt:
            async with AsyncSession(get_async_engine()) as session:
                # Get all submitted models and their provider names
                query = (
                    select(LanguageModel, Provider.name.label("provider_name"))  # type: ignore
                    .join(Provider, LanguageModel.provider_id == Provider.provider_id)  # type: ignore
                    .where(
                        LanguageModel.status == LanguageModelStatusEnum.SUBMITTED,
                        LanguageModel.deleted_at.is_(None),  # type: ignore
                    )
                )
                submitted_models = await session.execute(query)

            rows = list(submitted_models)
            print(f"Found {len(rows)} submitted models:")
            for model, provider_name in rows:
                print(f"    {provider_name:20}: {model.internal_name}")

            for model, _ in rows:
                await _verify_one_submitted_model(model)

            await revalidate_yupp_head_model_info()


async def verify_onboard_specific_model(model_id: UUID) -> None:
    """
    Verify and onboard a specific model.

    This function should be called to check and update the status of a specific model.
    It queries for the model, verifies it, and updates its status accordingly.
    """
    async for attempt in await _async_retry_decorator():
        with attempt:
            async with AsyncSession(get_async_engine()) as session:
                query = (
                    select(LanguageModel, Provider.name.label("provider_name"))  # type: ignore
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
                model, _ = row
                await _verify_one_submitted_model(model)

                await revalidate_yupp_head_model_info()
            else:
                log_dict = {"message": f"No submitted model found with id {model_id}", "model_id": model_id}
                logging.warning(json_dumps(log_dict))


async def _verify_one_submitted_model(model: LanguageModel) -> None:
    """
    Verify and update the status of a single model.

    Args:
        model (LanguageModel): The model to verify and update.
        provider_name (str): The name of the provider, this is mostly for display
    """
    async for attempt in await _async_retry_decorator():
        with attempt:
            async with AsyncSession(get_async_engine()) as session:
                try:
                    model_name = model.name  # just for the reference in error handling part
                    print(f"\nModel {model.name}: starting verifying submission")
                    is_inference_running, error_type, excerpt = await verify_inference_running(model)

                    if error_type:
                        await log_and_post(
                            ModelManagementStatus.OTHER_ERROR,
                            model_name,
                            f"{error_type}: {excerpt}",
                            level=ModelAlertLevel.ALERT,
                        )

                    if is_inference_running:
                        print(f"Model {model.name}: submission verification done: Success")
                        print(f">> [DB] updating model {model.name} status to ACTIVE")
                        model.status = LanguageModelStatusEnum.ACTIVE
                        model.modified_at = datetime.now(UTC)
                        session.add(model)
                        await session.commit()
                        await log_and_post(ModelManagementStatus.VALIDATED, model_name, level=ModelAlertLevel.ALERT)
                    else:
                        # reject if the model was submitted more than 3 days ago
                        print(f"Model {model.name}: submission verification done: Failed")
                        if (
                            model.created_at
                            and (datetime.now(UTC) - model.created_at.replace(tzinfo=UTC)).days > REJECT_AFTER_DAYS
                        ):
                            print(f"Model {model.name}: already failed for {REJECT_AFTER_DAYS} days, rejecting")
                            print(f">> [DB] updating model {model.name} status to REJECTED")
                            model.status = LanguageModelStatusEnum.REJECTED
                            model.modified_at = datetime.now(UTC)
                            session.add(model)
                            await session.commit()
                            await log_and_post(ModelManagementStatus.REJECTED, model_name, level=ModelAlertLevel.ALERT)
                        else:
                            print(f"Model {model.name}: not yet {REJECT_AFTER_DAYS} days, giving it more chances")
                            await log_and_post(
                                ModelManagementStatus.PENDING,
                                model_name,
                                level=ModelAlertLevel.NOTIFY,
                            )

                except Exception as e:
                    print(f"Model {model.name}: error while verifying submission: {e}")
                    await log_and_post(
                        ModelManagementStatus.VALIDATION_ERROR, model_name, str(e), level=ModelAlertLevel.ALERT
                    )
                    raise  # trigger retry


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_fixed(WAIT_TIME),
    retry=retry_if_exception_type((Exception, TimeoutError)),
)
async def revalidate_yupp_head_model_info() -> None:
    """Notify yupp-head when model ."""
    yupp_head_base_url = settings.YUPP_HEAD_APP_BASE_URL
    if not yupp_head_base_url:
        logging.error(json_dumps({"message": "Model Management: yupp-head base url not found for revalidation"}))
        return

    async with httpx.AsyncClient() as client:
        logging.info(
            json_dumps(
                {
                    "message": "Model Management: Revalidating yupp-head model info",
                    "yupp_head_base_url": yupp_head_base_url,
                }
            )
        )
        try:
            response = await client.get(
                f"{yupp_head_base_url}/api/models/revalidate",
                timeout=10.0,
                headers={"X-API-KEY": settings.X_API_KEY},
            )
            response.raise_for_status()
            logging.info(
                json_dumps(
                    {
                        "message": "Model Management: yupp-head model revalidation call succeeded",
                        "response": response.json(),
                    }
                )
            )
        except httpx.HTTPError as e:
            logging.error(
                json_dumps({"message": "Model Management: yupp-head model revalidation call failed", "error": str(e)})
            )
