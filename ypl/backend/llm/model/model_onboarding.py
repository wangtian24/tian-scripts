# Standard library imports
import logging
import os
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

# Third-party imports
import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)
from tenacity.asyncio import AsyncRetrying

# Local imports
from ypl.backend.config import settings
from ypl.backend.db import get_async_engine
from ypl.backend.llm.provider.provider_clients import get_provider_client
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import StopWatch
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, Provider

# Constants
MMLU_PRO_THRESHOLD = 50
DOWNLOADS_THRESHOLD = 1000
LIKES_THRESHOLD = 100
REJECT_AFTER_DAYS = 3
MAX_RETRIES = 3
WAIT_TIME = 5
INFERENCE_TIMEOUT = 60
INFERENCE_VERIFICATION_PROMPT = "What is the capital of Odisha?"

BILLING_ERROR_KEYWORDS = [
    "quota",
    "plan",
    "billing",
    "insufficient",
    "exceeded",
    "limit",
    "payment",
    "subscription",
    "credit",
    "balance",
    "access",
    "unauthorized",
]

# Replace the existing client caches with a single cache
provider_clients: dict[str, Any] = {}


def _contains_billing_error_keywords(error_message: str) -> tuple[bool, str]:
    for keyword in BILLING_ERROR_KEYWORDS:
        if keyword.lower() in error_message.lower():
            # Find index of first match
            match_idx = error_message.lower().find(keyword.lower())
            # Extract 30 chars before and after, handling string bounds
            start = max(0, match_idx - 30)
            end = min(len(error_message), match_idx + len(keyword) + 30)
            return True, error_message[start:end]
    return False, ""


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

            await _revalidate_yupp_head_model_info()


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

                await _revalidate_yupp_head_model_info()
            else:
                log_dict = {"message": f"No submitted model found with id {model_id}", "model_id": model_id}
                logging.warning(json_dumps(log_dict))


class ModelManagementStatus(Enum):
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    DEACTIVATED = "DEACTIVATED"
    BILLING_ERROR = "BILLING_ERROR"
    NOTIFY_NOT_RUNNING = "NOTIFY_NOT_RUNNING"
    NOTIFY_RECOVERED = "NOTIFY_RECOVERED"
    NOTIFY_MORE_CHANCE = "NOTIFY_MORE_CHANCE"
    ERROR = "FAILED"


MODEL_MANAGEMENT_STATUS_MESSAGES = {
    ModelManagementStatus.VALIDATED: "Set to ACTIVE after having been validated successfully",
    ModelManagementStatus.REJECTED: f"Set to REJECTED due to not being validated after {REJECT_AFTER_DAYS} days",
    ModelManagementStatus.DEACTIVATED: "Set to INACTIVE due to consecutive inference failures",
    ModelManagementStatus.BILLING_ERROR: "Billing error detected",
    ModelManagementStatus.NOTIFY_NOT_RUNNING: "Not running on the provider's endpoint, please investigate",
    ModelManagementStatus.NOTIFY_RECOVERED: "Recovered and is now running again on the provider's endpoint",
    ModelManagementStatus.NOTIFY_MORE_CHANCE: f"Still more chances for verification, not yet {REJECT_AFTER_DAYS} days",
    ModelManagementStatus.ERROR: "Error occurred while validating model",
}


async def _log_and_post(status: ModelManagementStatus, model_name: str, extra_msg: str | None = None) -> None:
    log_dict = {
        "message": f"Model Management: [{model_name}] - {MODEL_MANAGEMENT_STATUS_MESSAGES[status]}",
        "details": extra_msg,
    }
    if status == ModelManagementStatus.ERROR:
        logging.error(json_dumps(log_dict))
    else:
        logging.info(json_dumps(log_dict))

    print(f">> [log/slack] Model {model_name}: {MODEL_MANAGEMENT_STATUS_MESSAGES[status]} - {extra_msg or ''}")

    slack_msg = f"Model {model_name}: {MODEL_MANAGEMENT_STATUS_MESSAGES[status]} \n {extra_msg or ''} \n"
    f"[Environment: ]{os.environ.get('ENVIRONMENT')}]"

    await post_to_slack(slack_msg)


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
                    is_inference_running, has_billing_error, excerpt = await _verify_inference_running(model)

                    if has_billing_error:
                        await _log_and_post(ModelManagementStatus.BILLING_ERROR, model_name, excerpt)

                    if is_inference_running:
                        print(f"Model {model.name}: submission verification done: Success")
                        print(f">> [DB] updating model {model.name} status to ACTIVE")
                        model.status = LanguageModelStatusEnum.ACTIVE
                        model.modified_at = datetime.now(UTC)
                        session.add(model)
                        await session.commit()
                        await _log_and_post(ModelManagementStatus.VALIDATED, model_name)
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
                            await _log_and_post(ModelManagementStatus.REJECTED, model_name)
                        else:
                            print(f"Model {model.name}: not yet {REJECT_AFTER_DAYS} days, giving it more chances")
                            await _log_and_post(ModelManagementStatus.NOTIFY_MORE_CHANCE, model_name)

                except Exception as e:
                    print(f"Model {model.name}: error while verifying submission: {e}")
                    await _log_and_post(ModelManagementStatus.ERROR, model_name, str(e))
                    raise  # trigger retry


INFERENCE_ATTEMPTS = 3


@retry(stop=stop_after_attempt(INFERENCE_ATTEMPTS), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
async def check_inference_with_retries(client: BaseChatModel, model: LanguageModel) -> bool:
    attempt_number = check_inference_with_retries.statistics["attempt_number"]  # type: ignore
    log_dict = {
        "message": f"Model {model.name}: check inference, attempt {attempt_number}/{INFERENCE_ATTEMPTS}",
        "attempt": attempt_number,
    }
    logging.info(json_dumps(log_dict))

    stopwatch = StopWatch()
    try:
        results = client.invoke(INFERENCE_VERIFICATION_PROMPT)
        is_inference_running = results is not None and results.content is not None
        print(f"... response: {results.content if results and results.content else '[no response]'}")

        stopwatch.end(f"latency_{model.name}")
        log_dict = {
            "message": f"Attempt {attempt_number}/{INFERENCE_ATTEMPTS} completed for model {model.name}",
            "attempt": attempt_number,
            "model": f"{model.name}",
            "success": is_inference_running,
            "latency": stopwatch.get_splits(),
        }
        logging.info(json_dumps(log_dict))
        # If the inference failed and it's not the last attempt, raise an exception to trigger a retry.
        if not is_inference_running and attempt_number < INFERENCE_ATTEMPTS:
            raise Exception(f"Inference failed for model {model.name}")
        return is_inference_running

    except Exception as e:
        print(f"Error in check_inference_with_retries: {e}")
        log_dict = {
            "message": f"Model Management: Attempt {attempt_number}/{INFERENCE_ATTEMPTS} failed for model {model.name}",
            "attempt": attempt_number,
            "model": f"{model.name}",
            "error": str(e),
        }
        logging.error(json_dumps(log_dict))
        raise


async def _verify_inference_running(model: LanguageModel) -> tuple[bool, bool, str | None]:
    """
    Verify if the model is running on the provider's endpoint.

    Returns:
        tuple[bool, bool]: (is_inference_running, has_billing_error)
    """
    try:
        chat_model_client = await get_provider_client(model.internal_name, include_all_models=True)
        if not chat_model_client:
            raise Exception(f"Model {model.name} not found from any active provider")

        is_inference_running = await check_inference_with_retries(client=chat_model_client, model=model)

        return is_inference_running, False, None  # No billing error

    except Exception as e:
        print(f"Error in _verify_inference_running: {e}")
        log_dict = {
            "message": "Model Management: All inference attempts failed for model",
            "model_name": model.name,
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))

        has_billing_error, excerpt = _contains_billing_error_keywords(str(e))
        if has_billing_error:
            log_dict = {
                "message": f"Model Management: Potential billing error detected: ... {excerpt} ...",
                "model_name": model.name,
            }
            logging.error(json_dumps(log_dict))

        return False, has_billing_error, excerpt


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_fixed(WAIT_TIME),
    retry=retry_if_exception_type((Exception, TimeoutError)),
)
async def _revalidate_yupp_head_model_info() -> None:
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
