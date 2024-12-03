import logging
import os
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

import ypl.db.all_models  # noqa: F401
from ypl.backend.llm.error_logger import DatabaseLanguageModelErrorLogger, DefaultLanguageModelErrorLogger
from ypl.backend.llm.model.model import (
    LanguageModelStruct,
    create_model,
    delete_model,
    get_model_details,
    get_models,
    update_model,
)
from ypl.backend.llm.model.model_onboarding import verify_onboard_specific_model
from ypl.backend.llm.routing.route_data_type import InstantaneousLanguageModelStatistics, LanguageModelStatistics
from ypl.backend.llm.running_statistics import RunningStatisticsTracker
from ypl.backend.llm.utils import post_to_slack
from ypl.backend.utils.json import json_dumps
from ypl.db.language_models import (
    LanguageModel,
    LanguageModelResponseStatus,
    LanguageModelResponseStatusEnum,
    LanguageModelStatusEnum,
    LicenseEnum,
)

router = APIRouter()


async def async_verify_onboard_specific_models(model_id: UUID) -> None:
    try:
        await verify_onboard_specific_model(model_id)
    except Exception as e:
        log_dict = {
            "message": f"Error in verify_onboard_specific_model for model_id {model_id}: {str(e)}",
        }
        logging.exception(json_dumps(log_dict))


@router.post("/models", response_model=str)
async def create_model_route(model: LanguageModel, background_tasks: BackgroundTasks) -> str:
    try:
        model_id = create_model(model)
        background_tasks.add_task(async_verify_onboard_specific_models, model_id)
        background_tasks.add_task(
            post_to_slack,
            f"Environment {os.environ.get('ENVIRONMENT')} - Model {model.name} ({model.internal_name}) "
            "submitted for validation.",
        )
        return str(model_id)
    except Exception as e:
        log_dict = {
            "message": f"Error creating model - {str(e)}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models", response_model=list[LanguageModelStruct])
async def read_models_route(
    name: str | None = Query(None),  # noqa: B008
    licenses: list[LicenseEnum] | None = Query(None),  # noqa: B008
    family: str | None = Query(None),  # noqa: B008
    statuses: list[LanguageModelStatusEnum] = Query(None),  # noqa: B008
    creator_user_id: str | None = Query(None),  # noqa: B008
    exclude_deleted: bool = Query(True),  # noqa: B008
) -> list[LanguageModelStruct]:
    try:
        params = locals()
        return get_models(**params)
    except Exception as e:
        log_dict = {
            "message": f"Error getting models - {str(e)}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/model/{model_id}", response_model=LanguageModelStruct | None)
async def read_model_route(model_id: str) -> LanguageModelStruct | None:
    try:
        model = get_model_details(model_id)
        return model
    except Exception as e:
        log_dict = {
            "message": f"Error getting model - {str(e)}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.patch("/models/{model_id}", response_model=LanguageModelStruct)
async def update_model_route(model_id: str, updated_model: LanguageModel) -> LanguageModelStruct:
    try:
        return update_model(model_id, updated_model)
    except Exception as e:
        log_dict = {
            "message": f"Error updating model - {str(e)}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/models/{model_id}", status_code=204)
async def delete_model_route(model_id: str) -> None:
    try:
        delete_model(model_id)
    except Exception as e:
        log_dict = {
            "message": f"Error deleting model - {str(e)}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.patch("/models/{model_id}/running_statistics", response_model=LanguageModelStatistics)
async def update_model_running_statistics_route(
    model_id: str, statistics: InstantaneousLanguageModelStatistics
) -> LanguageModelStatistics:
    try:
        return await RunningStatisticsTracker.get_instance().update_statistics(model_id, statistics)
    except Exception as e:
        log_dict = {
            "message": f"Error updating model running statistics - {str(e)}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models/{model_id}/running_statistics", response_model=LanguageModelStatistics)
async def get_model_running_statistics_route(model_id: str) -> LanguageModelStatistics:
    try:
        return await RunningStatisticsTracker.get_instance().get_statistics(model_id)
    except Exception as e:
        log_dict = {
            "message": f"Error getting model running statistics - {str(e)}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


class LanguageModelErrorPayload(BaseModel):
    error_type: LanguageModelResponseStatusEnum
    error_message: str | None = None
    http_response_code: int | None = None


@router.post("/models/{model_id}/errors", response_model=None)
def log_model_error_route(model_id: str, payload: LanguageModelErrorPayload) -> None:
    try:
        error = LanguageModelResponseStatus(
            language_model_id=model_id,
            **payload.model_dump(),
        )
        DefaultLanguageModelErrorLogger().log(error)
        DatabaseLanguageModelErrorLogger().log(error)
    except Exception as e:
        log_dict = {"message": f"Error logging model error - {str(e)}"}
        logging.exception(json_dumps(log_dict))

        raise HTTPException(status_code=500, detail=str(e)) from e
