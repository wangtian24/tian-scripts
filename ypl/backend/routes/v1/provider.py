import json
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from ypl.backend.llm.provider.provider import (
    ProviderStruct,
    create_provider,
    delete_provider,
    get_provider_details,
    get_providers,
    update_provider,
)
from ypl.db.language_models import Provider

router = APIRouter()


@router.post("/providers", response_model=UUID)
async def create_provider_route(provider: Provider) -> UUID:
    try:
        provider_id = await create_provider(provider)
        return provider_id
    except Exception as e:
        log_dict = {
            "message": f"Error creating provider - {str(e)}",
        }
        logging.exception(json.dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/providers", response_model=list[ProviderStruct])
async def read_providers_route(
    name: str | None = Query(None),  # noqa: B008
    is_active: bool = Query(True),  # noqa: B008
    exclude_deleted: bool = Query(True),  # noqa: B008
) -> list[ProviderStruct]:
    try:
        params = locals()
        return await get_providers(**params)
    except Exception as e:
        log_dict = {
            "message": f"Error getting providers - {str(e)}",
        }
        logging.exception(json.dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/provider/{provider_id}", response_model=ProviderStruct | None)
async def read_provider_route(provider_id: UUID) -> ProviderStruct | None:
    try:
        provider = await get_provider_details(provider_id)
        return provider
    except Exception as e:
        log_dict = {
            "message": f"Error getting provider - {str(e)}",
        }
        logging.exception(json.dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.patch("/providers/{provider_id}", response_model=ProviderStruct)
async def update_provider_route(provider_id: UUID, updated_provider: Provider) -> ProviderStruct:
    try:
        return await update_provider(provider_id, updated_provider)
    except Exception as e:
        log_dict = {
            "message": f"Error updating provider - {str(e)}",
        }
        logging.exception(json.dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/providers/{provider_id}", status_code=204)
async def delete_provider_route(provider_id: UUID) -> None:
    try:
        await delete_provider(provider_id)
    except Exception as e:
        log_dict = {
            "message": f"Error deleting provider - {str(e)}",
        }
        logging.exception(json.dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
