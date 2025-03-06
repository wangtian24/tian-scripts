import logging
import os
from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from tenacity import retry, stop_after_attempt, wait_exponential
from ypl.backend.llm.provider.provider_clients import get_provider_client
from ypl.backend.llm.utils import post_to_slack_channel
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.utils import StopWatch
from ypl.db.language_models import LanguageModel

REJECT_AFTER_DAYS = 3
INFERENCE_TIMEOUT = 60
INFERENCE_VERIFICATION_PROMPT = "What is the capital of Odisha?"
INFERENCE_ATTEMPTS = 3


class ModelManagementStatus(Enum):
    # -- for onboarding --
    VALIDATED = "VALIDATED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    # -- for periodical validation --
    PROBATION = "PROBATION"
    RECOVERED = "NOTIFY_RECOVERED"
    DEACTIVATED = "DEACTIVATED"
    NOT_ENOUGH_TRAFFIC = "NOTIFY_NOT_ENOUGH_TRAFFIC"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    # -- shared --
    OTHER_ERROR = "DETECTED_ERROR"


MODEL_MANAGEMENT_STATUS_MESSAGES = {
    # -- for onboarding --
    ModelManagementStatus.VALIDATED: (
        "Onboarding: New model submission verification success. Set to ACTIVE after having been validated successfully"
    ),
    ModelManagementStatus.REJECTED: (
        f"Onboarding: New model submission verification failed. "
        f"Set to REJECTED due to not being validated after {REJECT_AFTER_DAYS} days"
    ),
    ModelManagementStatus.PENDING: (
        f"Onboarding: New model submission verification failed. "
        f"Additional validation tests will be conducted until day {REJECT_AFTER_DAYS}."
    ),
    ModelManagementStatus.VALIDATION_ERROR: ("Error occurred while validating model"),
    # -- for periodical validation --
    ModelManagementStatus.PROBATION: ("Validation: Failed validation, set to PROBATION due to high error rate"),
    ModelManagementStatus.DEACTIVATED: ("Validation: Failed probation, set to INACTIVE after consecutive failures"),
    ModelManagementStatus.RECOVERED: ("Validation: Exited probation, set to ACTIVE again after consecutive successes"),
    ModelManagementStatus.NOT_ENOUGH_TRAFFIC: ("Validation: Not enough traffic to decide, will continue monitoring"),
    ModelManagementStatus.INFERENCE_ERROR: ("Validation: Inference error"),
    # -- shared --
    ModelManagementStatus.OTHER_ERROR: ("Detected error"),
}


class ModelAlertLevel(Enum):
    LOG_ONLY = 0  # Log only
    NOTIFY = 10  # Log, send to #alert-model-management
    ALERT = 20  # Log, send to #alert-model-management and #alert-backend


async def log_and_post(
    status: ModelManagementStatus,
    model_name: str,
    extra_msg: str | None = None,
    level: ModelAlertLevel = ModelAlertLevel.NOTIFY,
) -> None:
    log_dict = {
        "message": f"Model Management: [{model_name}] - {MODEL_MANAGEMENT_STATUS_MESSAGES[status]}",
        "details": extra_msg,
    }
    if status == ModelManagementStatus.VALIDATION_ERROR:
        logging.error(json_dumps(log_dict))
    else:
        logging.info(json_dumps(log_dict))

    print(f">> [log/slack] Model {model_name}: {MODEL_MANAGEMENT_STATUS_MESSAGES[status]} - {extra_msg or ''}")

    env = os.environ.get("ENVIRONMENT") or "unknown"

    # Clean up and escape quotes in extra_msg to ensure it's safe for logging and Slack
    if level.value >= ModelAlertLevel.NOTIFY.value:
        slack_msg = (
            f"[{env}] Model *{model_name}*: " f"{MODEL_MANAGEMENT_STATUS_MESSAGES[status]}\n {extra_msg or ''} \n"
        )
        await post_to_slack_channel(slack_msg, "#alert-model-management")
    if level.value >= ModelAlertLevel.ALERT.value:
        await post_to_slack_channel(
            slack_msg, "#alert-backend" if env.lower() == "production" else "#alert-backend-staging"
        )


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


class ModelErrorType(Enum):
    CONTEXT_LENGTH = "CONTEXT_LENGTH"
    RATE_LIMIT = "RATE_LIMIT"
    BILLING = "BILLING"
    UNKNOWN = "UNKNOWN"


ERROR_KEYWORDS_MAP = {
    # detect in this order
    ModelErrorType.CONTEXT_LENGTH: [
        "payload size exceeds",
        "context length",
        "input length",
        "max tokens",
    ],
    ModelErrorType.RATE_LIMIT: ["rate limit"],
    ModelErrorType.BILLING: [
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
        "purchase",
        "exhausted",
    ],
}


def contains_error_keywords(error_message: str) -> tuple[ModelErrorType | None, str | None]:
    for error_type, keywords in ERROR_KEYWORDS_MAP.items():
        for keyword in keywords:
            if keyword.lower() in error_message.lower():
                # Find index of first match
                match_idx = error_message.lower().replace("_", " ").find(keyword.lower())
                # Extract excerpts
                start = max(0, match_idx - 100)
                end = min(len(error_message), match_idx + len(keyword) + 100)
                return error_type, error_message[start:end]
    return ModelErrorType.UNKNOWN, error_message[:200]


async def verify_inference_running(model: LanguageModel) -> tuple[bool, ModelErrorType | None, str | None]:
    """
    Verify if the model is running on the provider's endpoint.

    Returns:
        tuple[bool, bool]: (is_inference_running, error_type, excerpt)
    """
    try:
        chat_model_client = await get_provider_client(internal_name=model.internal_name, include_all_models=True)
        if not chat_model_client:
            raise Exception(f"Model {model.name} not found from any active provider")

        print(f"... {model.name}")
        is_inference_running = await check_inference_with_retries(client=chat_model_client, model=model)

        return is_inference_running, None, None  # No billing error

    except Exception as e:
        print(f"Error in _verify_inference_running: {e}")
        log_dict = {
            "message": "Model Management: All inference attempts failed for model",
            "model_name": model.name,
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))

        error_type, excerpt = contains_error_keywords(str(e))
        if error_type:
            log_dict = {
                "message": f"Model Management: Potential {error_type.value} error detected: ... {excerpt} ...",
                "model_name": model.name,
            }
            logging.error(json_dumps(log_dict))

        return False, error_type, excerpt
