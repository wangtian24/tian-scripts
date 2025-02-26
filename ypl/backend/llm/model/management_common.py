import logging
import os
from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from tenacity import retry, stop_after_attempt, wait_exponential
from ypl.backend.llm.provider.provider_clients import get_provider_client
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import StopWatch
from ypl.db.language_models import LanguageModel

REJECT_AFTER_DAYS = 3
INFERENCE_TIMEOUT = 60
INFERENCE_VERIFICATION_PROMPT = "What is the capital of Odisha?"
INFERENCE_ATTEMPTS = 3


class ModelManagementStatus(Enum):
    ONBOARDING_VALIDATED = "VALIDATED"
    ONBOARDING_REJECTED = "REJECTED"
    ONBOARDING_NOTIFY_MORE_CHANCE = "NOTIFY_MORE_CHANCE"
    VALIDATION_DEACTIVATED = "DEACTIVATED"
    VALIDATION_BILLING_ERROR = "BILLING_ERROR"
    VALIDATION_NOTIFY_NOT_RUNNING = "NOTIFY_NOT_RUNNING"
    VALIDATION_NOTIFY_RECOVERED = "NOTIFY_RECOVERED"
    ERROR = "FAILED"


MODEL_MANAGEMENT_STATUS_MESSAGES = {
    ModelManagementStatus.ONBOARDING_VALIDATED: (
        "New model submission verification success. Set to ACTIVE after having been validated successfully"
    ),
    ModelManagementStatus.ONBOARDING_REJECTED: (
        f"New model submission verification failed. "
        f"Set to REJECTED due to not being validated after {REJECT_AFTER_DAYS} days"
    ),
    ModelManagementStatus.ONBOARDING_NOTIFY_MORE_CHANCE: (
        f"New model submission verification failed. Still more chances, not yet {REJECT_AFTER_DAYS} days"
    ),
    ModelManagementStatus.VALIDATION_DEACTIVATED: (
        "Inference validation failed. Set to INACTIVE due to consecutive inference failures"
    ),
    ModelManagementStatus.VALIDATION_BILLING_ERROR: ("Billing error detected"),
    ModelManagementStatus.VALIDATION_NOTIFY_NOT_RUNNING: (
        "Inference validation failed. Not running on the provider's endpoint, please investigate"
    ),
    ModelManagementStatus.VALIDATION_NOTIFY_RECOVERED: (
        "Inference validation success. Recovered and now running again on the provider's endpoint"
    ),
    ModelManagementStatus.ERROR: ("Error occurred while validating model"),
}


async def log_and_post(status: ModelManagementStatus, model_name: str, extra_msg: str | None = None) -> None:
    log_dict = {
        "message": f"Model Management: [{model_name}] - {MODEL_MANAGEMENT_STATUS_MESSAGES[status]}",
        "details": extra_msg,
    }
    if status == ModelManagementStatus.ERROR:
        logging.error(json_dumps(log_dict))
    else:
        logging.info(json_dumps(log_dict))

    print(f">> [log/slack] Model {model_name}: {MODEL_MANAGEMENT_STATUS_MESSAGES[status]} - {extra_msg or ''}")

    slack_msg = f"*Model {model_name}: {MODEL_MANAGEMENT_STATUS_MESSAGES[status]} *\n {extra_msg or ''} \n"
    f"[Environment: ]{os.environ.get('ENVIRONMENT')}]"

    await post_to_slack(slack_msg)


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


def contains_billing_error_keywords(error_message: str) -> tuple[bool, str]:
    for keyword in BILLING_ERROR_KEYWORDS:
        if keyword.lower() in error_message.lower():
            # Find index of first match
            match_idx = error_message.lower().find(keyword.lower())
            # Extract 30 chars before and after, handling string bounds
            start = max(0, match_idx - 50)
            end = min(len(error_message), match_idx + len(keyword) + 50)
            return True, error_message[start:end]
    return False, ""


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

        has_billing_error, excerpt = contains_billing_error_keywords(str(e))
        if has_billing_error:
            log_dict = {
                "message": f"Model Management: Potential billing error detected: ... {excerpt} ...",
                "model_name": model.name,
            }
            logging.error(json_dumps(log_dict))

        return False, has_billing_error, excerpt
