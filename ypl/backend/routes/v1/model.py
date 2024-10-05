import logging

from fastapi import APIRouter, HTTPException, Query

from ypl.backend.llm.model.model import (
    LanguageModelStruct,
    create_model,
    delete_model,
    get_model_details,
    get_models,
    update_model,
)
from ypl.db.language_models import LanguageModel, LanguageModelStatusEnum, LicenseEnum

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
router = APIRouter()


@router.post("/models", response_model=str)
async def create_model_route(model: LanguageModel) -> str:
    try:
        model_id = create_model(model)
        return str(model_id)
    except Exception as e:
        logger.exception("Error creating model - %s", str(e))
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
        logger.exception("Error getting models - %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/model/{model_id}", response_model=LanguageModelStruct | None)
async def read_model_route(model_id: str) -> LanguageModelStruct | None:
    try:
        model = get_model_details(model_id)
        return model
    except Exception as e:
        logger.exception("Error getting model - %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.patch("/models/{model_id}", response_model=LanguageModelStruct)
async def update_model_route(model_id: str, updated_model: LanguageModel) -> LanguageModelStruct:
    try:
        return update_model(model_id, updated_model)
    except Exception as e:
        logger.exception("Error updating model - %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/models/{model_id}", status_code=204)
async def delete_model_route(model_id: str) -> None:
    try:
        delete_model(model_id)
    except Exception as e:
        logger.exception("Error deleting model - %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
