import logging
import os
import uuid
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import UUID4, BaseModel
from sqlalchemy import func

from ypl.backend.llm.error_logger import DatabaseLanguageModelStatusLogger, DefaultLanguageModelStatusLogger
from ypl.backend.llm.model.model import (
    LanguageModelStruct,
    ModelTaxonomyQuery,
    ModelTaxonomyResponse,
    create_model,
    create_model_taxonomy,
    delete_model,
    get_model_details,
    get_model_taxonomies,
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
    LanguageModelTaxonomy,
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


class CreateLanguageModelRequest(BaseModel):
    # TODO(Tian): the 'name' is kep for backward compatibility, we will ask user directly for internal name
    # in the future, and the name will be auto-set from provider name and internal name.
    name: str
    internal_name: str | None = None  # TOOD(Tian): will become required later
    label: str
    creator_user_id: str
    provider_id: str
    license: str | None = None
    family: str | None = None
    avatar_url: str | None = None
    parameter_count: int | None = None
    context_window_tokens: int | None = None
    knowledge_cutoff_date: datetime | None = None
    model_publisher: str | None = None  # not in language_models_table
    model_class: str | None = None
    model_version: str | None = None
    model_release: str | None = None
    is_pro: bool | None = None
    is_strong: bool | None = None
    is_live: bool | None = None
    external_model_info_url: str | None = None
    organization_id: UUID4 | None = None
    supported_attachment_mime_types: list[str] | None = None  # a list of mime type strings
    parameters: dict[str, Any] | None = None  # json blob


@router.post("/models", response_model=str)
async def create_model_route(create_req: CreateLanguageModelRequest, background_tasks: BackgroundTasks) -> str:
    try:
        # construct a LanguageModel object from the CreateLanguageModelRequest object
        # don't directly use an ORM object as API request argument
        model = LanguageModel(
            created_at=func.now(),
            language_model_id=uuid.uuid4(),
            name=create_req.name,
            internal_name=create_req.internal_name,
            label=create_req.label,
            license=create_req.license or "unknown",
            family=create_req.family,
            avatar_url=create_req.avatar_url,
            parameter_count=create_req.parameter_count,
            context_window_tokens=create_req.context_window_tokens,
            knowledge_cutoff_date=create_req.knowledge_cutoff_date,
            status=LanguageModelStatusEnum.SUBMITTED,
            creator_user_id=create_req.creator_user_id,
            provider_id=create_req.provider_id,
            model_publisher=create_req.model_publisher or "UNKNOWN",
            model_class=create_req.model_class,
            model_version=create_req.model_version,
            model_release=create_req.model_release,
            is_pro=create_req.is_pro,
            is_strong=create_req.is_strong,
            is_live=create_req.is_live,
            external_model_info_url=create_req.external_model_info_url,
            organization_id=create_req.organization_id,
            supported_attachment_mime_types=create_req.supported_attachment_mime_types,
            parameters=create_req.parameters,
        )

        model_id = await create_model(model)
        background_tasks.add_task(async_verify_onboard_specific_models, model_id)
        background_tasks.add_task(
            post_to_slack,
            f"[{os.environ.get('ENVIRONMENT')}] - Model {create_req.name} submitted for validation.",
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
        return await get_models(**params)
    except Exception as e:
        log_dict = {
            "message": f"Error getting models - {str(e)}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/model/{model_id}", response_model=LanguageModelStruct | None)
async def read_model_route(model_id: str) -> LanguageModelStruct | None:
    try:
        model = await get_model_details(model_id)
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
        return await update_model(model_id, updated_model)
    except Exception as e:
        log_dict = {
            "message": f"Error updating model - {str(e)}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/models/{model_id}", status_code=204)
async def delete_model_route(model_id: str) -> None:
    try:
        await delete_model(model_id)
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


class LanguageModelStatusPayload(BaseModel):
    status_type: LanguageModelResponseStatusEnum
    status_message: str | None = None
    http_response_code: int | None = None


@router.post("/models/{model_id}/statuses", response_model=None)
async def log_model_status_route(model_id: str, payload: LanguageModelStatusPayload) -> None:
    try:
        status = LanguageModelResponseStatus(
            language_model_id=model_id,
            **payload.model_dump(),
        )
        DefaultLanguageModelStatusLogger().log(status)
        DatabaseLanguageModelStatusLogger().log(status)
    except Exception as e:
        log_dict = {"message": f"Error logging model status - {str(e)}"}
        logging.exception(json_dumps(log_dict))

        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/models/taxonomy/create", response_model=str)
async def create_model_taxonomy_route(model_taxonomy: LanguageModelTaxonomy, background_tasks: BackgroundTasks) -> str:
    try:
        model_taxonomy_id = await create_model_taxonomy(model_taxonomy)
        if model_taxonomy_id is None:
            raise HTTPException(status_code=409, detail="Taxonomy already exists")
        background_tasks.add_task(
            post_to_slack,
            f"[{os.environ.get('ENVIRONMENT')}] - Created new model taxonomy node: "
            f"{model_taxonomy.model_publisher}/{model_taxonomy.model_family}/{model_taxonomy.model_class}/"
            f"{model_taxonomy.model_version}/{model_taxonomy.model_release}",
        )
        return str(model_taxonomy_id)
    except Exception as e:
        log_dict = {
            "message": f"Error creating model taxonomy - {str(e)}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/models/taxonomy/query", response_model=list[ModelTaxonomyResponse])
async def get_model_taxonomy_route(query: ModelTaxonomyQuery) -> list[ModelTaxonomyResponse]:
    try:
        return await get_model_taxonomies(query)
    except Exception as e:
        log_dict = {
            "message": f"Error getting model taxonomies - {str(e)}",
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
