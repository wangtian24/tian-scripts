import logging

from fastapi import APIRouter, HTTPException

from ypl.backend.llm.model.model import (
    LanguageModelResponseBody,
    create_model,
    delete_model,
    get_leaderboard_model_details,
    get_leaderboard_models,
    update_model,
)
from ypl.db.language_models import LanguageModel

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


@router.get("/models", response_model=list[LanguageModelResponseBody])
async def read_models_route() -> list[LanguageModelResponseBody]:
    try:
        return get_leaderboard_models()
    except Exception as e:
        logger.exception("Error getting models - %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/model/{model_id}", response_model=LanguageModelResponseBody | None)
async def read_model_route(model_id: str) -> LanguageModelResponseBody | None:
    try:
        model = get_leaderboard_model_details(model_id)
        return model
    except Exception as e:
        logger.exception("Error getting model - %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.patch("/models/{model_id}", response_model=LanguageModelResponseBody)
async def update_model_route(model_id: str, updated_model: LanguageModel) -> LanguageModelResponseBody:
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
