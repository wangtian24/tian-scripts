import logging
from collections import defaultdict
from datetime import datetime, timedelta

from pydantic import BaseModel
from sqlalchemy import desc, func, select, update
from sqlalchemy.exc import DatabaseError, OperationalError
from tenacity import after_log, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from ypl.backend.db import get_async_session
from ypl.backend.llm.model.management_common import (
    ModelAlertLevel,
    ModelManagementStatus,
    log_and_post,
    verify_inference_running,
)
from ypl.db.language_models import (
    LanguageModel,
    LanguageModelResponseStatus,
    LanguageModelResponseStatusEnum,
    LanguageModelStatusEnum,
    Provider,
)

# The status types reserved for model management
INFERENCE_STATUS_TYPES: list[LanguageModelResponseStatusEnum] = [
    LanguageModelResponseStatusEnum.INFERENCE_SUCCEEDED,
    LanguageModelResponseStatusEnum.INFERENCE_FAILED,
]

DATABASE_ATTEMPTS = 3
DATABASE_WAIT_TIME_SECONDS = 0.1
DATABASE_RETRY_IF_EXCEPTION_TYPE = (OperationalError, DatabaseError)

# only check models with at least one activity in this recent time window
VALIDATION_ACTIVITY_WINDOW = timedelta(minutes=15)  # should be same as cron job interval

# short/long term error rate calculation windows
VALIDATION_CHECK_SHORT_WINDOW = timedelta(minutes=60)
VALIDATION_CHECK_LONG_WINDOW = timedelta(hours=6)

# Minimum number of requests needed to calculate the error rate. If there's too few traffic,
# We don't take any action. (this validation check will also generate traffic though)
MIN_REQS_FOR_SHORT_TERM = 10
MIN_REQS_FOR_LONG_TERM = 30

# If the error rates goes above these thresholds, we will put the model in probation.
MAX_SHORT_TERM_ERROR_RATE = 0.4
MAX_LONG_TERM_ERROR_RATE = 0.2

# How many consecutive failures or successes before we exit probation.
PROBATION_EXIT_SUCCESS_REQS = 3
PROBATION_EXIT_FAILURE_REQS = 18


@retry(
    stop=stop_after_attempt(DATABASE_ATTEMPTS),
    wait=wait_fixed(DATABASE_WAIT_TIME_SECONDS),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type(DATABASE_RETRY_IF_EXCEPTION_TYPE),
)
async def _get_active_and_probation_models() -> list[LanguageModel]:
    """
    get all active and probation models that has any activity in the past VALIDATION_CHECK_INTERVAL,
    return them in two lists: active_models and probation_models
    """
    async with get_async_session() as session:
        query = (
            select(LanguageModel)
            .join(Provider, Provider.provider_id == LanguageModel.provider_id)  # type: ignore
            .join(
                LanguageModelResponseStatus,
                LanguageModelResponseStatus.language_model_id == LanguageModel.language_model_id,  # type: ignore
            )
            .where(
                LanguageModel.status.in_([LanguageModelStatusEnum.ACTIVE, LanguageModelStatusEnum.PROBATION]),  # type: ignore
                LanguageModel.deleted_at.is_(None),  # type: ignore
                Provider.deleted_at.is_(None),  # type: ignore
                Provider.is_active.is_(True),  # type: ignore
                LanguageModelResponseStatus.created_at > datetime.now() - VALIDATION_ACTIVITY_WINDOW,  # type: ignore
            )
            .distinct(LanguageModel.language_model_id)  # type: ignore
        )
        return list((await session.exec(query)).scalars().all())  # type: ignore


class ModelErrorRateInfo(BaseModel):
    short_term_error_rate: float
    long_term_error_rate: float


def _calc_error_rate(status_count_map: dict[str, defaultdict[str, int]], min_count: int) -> dict[str, float]:
    return {
        model_id: sum(v for c, v in error_counts.items() if not LanguageModelResponseStatusEnum(c).is_ok())
        / sum(error_counts.values())
        for model_id, error_counts in status_count_map.items()
        if sum(error_counts.values()) >= min_count
    }


async def _get_error_rate_map(window_start: datetime) -> dict[str, float]:
    """
    Gets error rates for all models in a specific time window ending now().
    """
    async with get_async_session() as session:
        query = (
            select(
                LanguageModel.name,
                LanguageModelResponseStatus.status_type,
                func.count(LanguageModelResponseStatus.status_type),  # type: ignore[arg-type]
            )  # type: ignore
            .where(
                LanguageModelResponseStatus.created_at > window_start,  # type: ignore
            )
            .join(LanguageModel)
            .group_by(
                LanguageModel.name,
                LanguageModelResponseStatus.status_type,
            )
        )
        results = (await session.exec(query)).all()

    error_map: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name, status_type, error_count in results:
        error_map[name][status_type] = error_count

    return _calc_error_rate(error_map, MIN_REQS_FOR_LONG_TERM)


async def _get_long_short_term_error_rate_map() -> dict[str, ModelErrorRateInfo]:
    now = datetime.now()
    short_term_error_rates = await _get_error_rate_map(now - VALIDATION_CHECK_SHORT_WINDOW)
    long_term_error_rates = await _get_error_rate_map(now - VALIDATION_CHECK_LONG_WINDOW)

    error_rate_info_map = {}
    for name in set(short_term_error_rates.keys()) | set(long_term_error_rates.keys()):
        error_rate_info_map[name] = ModelErrorRateInfo(
            short_term_error_rate=short_term_error_rates.get(name, 0.0),
            long_term_error_rate=long_term_error_rates.get(name, 0.0),
        )

    return error_rate_info_map


async def _test_inference_for_model(model: LanguageModel) -> None:
    """
    Do inference check on one model and write the status to DB
    """
    try:
        is_inference_running, error_type, excerpt = await verify_inference_running(model)

        await _insert_language_model_response_status(
            model,
            LanguageModelResponseStatusEnum.INFERENCE_SUCCEEDED
            if is_inference_running
            else LanguageModelResponseStatusEnum.INFERENCE_FAILED,
        )
        if not is_inference_running:
            await log_and_post(ModelManagementStatus.INFERENCE_ERROR, model.name, excerpt, level=ModelAlertLevel.NOTIFY)

        if error_type and datetime.now().minute < 10:  # alert at most once an hour (cron job runs every 15 minutes)
            await log_and_post(
                ModelManagementStatus.OTHER_ERROR,
                model.name,
                f"{error_type}: {excerpt}",
                level=ModelAlertLevel.ALERT,
            )
    except Exception as e:
        print(f"Model {model.name}: error during validation: {e}")
        await log_and_post(ModelManagementStatus.VALIDATION_ERROR, model.name, str(e), level=ModelAlertLevel.ALERT)


@retry(
    stop=stop_after_attempt(DATABASE_ATTEMPTS),
    wait=wait_fixed(DATABASE_WAIT_TIME_SECONDS),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type(DATABASE_RETRY_IF_EXCEPTION_TYPE),
)
async def _insert_language_model_response_status(
    model: LanguageModel, status_type: LanguageModelResponseStatusEnum
) -> None:
    print(f">> [DB] updating model {model.name} response status to {status_type}")
    async with get_async_session() as session:
        new_language_model_response_record = LanguageModelResponseStatus(
            language_model_id=model.language_model_id,
            status_type=status_type,
        )
        session.add(new_language_model_response_record)
        await session.commit()


async def _get_num_consecutive_state(
    model_id: str,
    target_status_type: LanguageModelResponseStatusEnum,
    time_delta: timedelta = VALIDATION_CHECK_LONG_WINDOW,
) -> int:
    # Get the number of consecutive state in the past time window
    async with get_async_session() as session:
        query = (
            select(LanguageModelResponseStatus.status_type)  # type: ignore
            .where(
                LanguageModelResponseStatus.language_model_id == model_id,
                LanguageModelResponseStatus.status_type.in_(INFERENCE_STATUS_TYPES),  # type: ignore
                (LanguageModelResponseStatus.created_at > datetime.now() - time_delta),  # type: ignore
            )
            .order_by(desc(LanguageModelResponseStatus.created_at))  # type: ignore
        )
        results = (await session.exec(query)).all()

        # Count the number of consecutive states starting from the most recent one.
        # if the target state is FAIL and the sequence is
        #         (now)->FAIL, FAIL, SUCCESS, FAIL->(past)
        # we will count 2 failures.
        num_consecutive_states = 0
        for status_type in results:
            if status_type == target_status_type:
                num_consecutive_states += 1
            else:
                break

        return num_consecutive_states


@retry(
    stop=stop_after_attempt(DATABASE_ATTEMPTS),
    wait=wait_fixed(DATABASE_WAIT_TIME_SECONDS),
    after=after_log(logging.getLogger(), logging.WARNING),
    retry=retry_if_exception_type(DATABASE_RETRY_IF_EXCEPTION_TYPE),
)
async def _update_model_status(model: LanguageModel, status: LanguageModelStatusEnum) -> None:
    print(f">> [DB] updating model {model.name} status to {status}")
    async with get_async_session() as session:
        print(f">> [DB] updating model {model.name} status to {status}")
        await session.exec(
            update(LanguageModel)
            .values(status=status)
            .where(LanguageModel.language_model_id == model.language_model_id)  # type: ignore
        )
        await session.commit()


async def _execute_validation_state_machine(model: LanguageModel, error_rate_info: ModelErrorRateInfo | None) -> None:
    """
    Execute the validation statemachine based on the error rate info.
        ACTIVE -> PROBATION if error rate is too high (short or long term)
        PROBATION -> INACTIVE if there are more than X consecutive failures
        PROBATION -> ACTIVE if there are more than Y consecutive successes
    """
    model_name = model.name

    print(f"\nExecuting state machine on model {model.name:60} {error_rate_info}")

    if model.status == LanguageModelStatusEnum.ACTIVE:
        if error_rate_info is not None:
            if error_rate_info.long_term_error_rate > MAX_LONG_TERM_ERROR_RATE:
                print("ACTIVE -> PROBATION")
                await _update_model_status(model, LanguageModelStatusEnum.PROBATION)  # ACTIVE -> PROBATION
                await log_and_post(
                    ModelManagementStatus.PROBATION,
                    model_name,
                    extra_msg=(
                        f"long term error rate = {error_rate_info.long_term_error_rate*100:.1f}% > "
                        f"threshold {MAX_LONG_TERM_ERROR_RATE*100:.1f}%"
                    ),
                    level=ModelAlertLevel.ALERT,
                )
            elif error_rate_info.short_term_error_rate > MAX_SHORT_TERM_ERROR_RATE:
                print("ACTIVE -> PROBATION")
                await _update_model_status(model, LanguageModelStatusEnum.PROBATION)  # ACTIVE -> PROBATION
                await log_and_post(
                    ModelManagementStatus.PROBATION,
                    model_name,
                    extra_msg=(
                        f"short term error rate = {error_rate_info.short_term_error_rate*100:.1f}% > "
                        f"threshold {MAX_SHORT_TERM_ERROR_RATE*100:.1f}%"
                    ),
                    level=ModelAlertLevel.ALERT,
                )
            else:
                print("ACTIVE -> ACTIVE")
        else:
            print("ACTIVE -> ACTIVE (not enough traffic)")
    elif model.status == LanguageModelStatusEnum.PROBATION:
        num_consecutive_success = await _get_num_consecutive_state(
            str(model.language_model_id), LanguageModelResponseStatusEnum.INFERENCE_SUCCEEDED
        )
        num_consecutive_failures = await _get_num_consecutive_state(
            str(model.language_model_id), LanguageModelResponseStatusEnum.INFERENCE_FAILED
        )
        if num_consecutive_success >= PROBATION_EXIT_SUCCESS_REQS:
            print("PROBATION -> ACTIVE")
            await _update_model_status(model, LanguageModelStatusEnum.ACTIVE)  # PROBATION -> ACTIVE
            await log_and_post(ModelManagementStatus.RECOVERED, model_name, level=ModelAlertLevel.ALERT)
        elif num_consecutive_failures >= PROBATION_EXIT_FAILURE_REQS:
            print("PROBATION -> INACTIVE")
            await _update_model_status(model, LanguageModelStatusEnum.INACTIVE)  # PROBATION -> INACTIVE
            await log_and_post(ModelManagementStatus.DEACTIVATED, model_name, level=ModelAlertLevel.ALERT)
        else:
            print("PROBATION -> PROBATION")


async def do_validate_active_models() -> None:
    """
    Validate active and probation models.

    This function should be run periodically to check and update the status of active models.
    It queries for active models, verifies them, and logs an error if any validation fails.
    """
    models = await _get_active_and_probation_models()

    # Print count of active and probation models
    active_models = [model for model in models if model.status == LanguageModelStatusEnum.ACTIVE]
    probation_models = [model for model in models if model.status == LanguageModelStatusEnum.PROBATION]
    print(
        f"-- Found {len(active_models)} ACTIVE models and {len(probation_models)} PROBATION models "
        f"in the past {VALIDATION_ACTIVITY_WINDOW}"
    )
    if len(probation_models) > 0:
        print(f"-- PROBATION models: {', '.join([model.name for model in probation_models])}")

    # Test inference for each model concurrently and write status to DB
    # Call test inference for each model one by one instead of concurrently
    for model in models:
        await _test_inference_for_model(model)

    # Collect all model's error rate info
    ls_error_rate_map = await _get_long_short_term_error_rate_map()

    # Print all model error rate info
    for name, error_info in ls_error_rate_map.items():
        print(
            f"{name:60} - "
            f"Short term: {error_info.short_term_error_rate*100:5.1f}%, "
            f"Long term: {error_info.long_term_error_rate*100:5.1f}%"
        )

    for model in models:
        await _execute_validation_state_machine(model, ls_error_rate_map.get(model.name))

    print("Done")
