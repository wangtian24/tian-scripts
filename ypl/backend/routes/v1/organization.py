import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from ypl.backend.llm.organization import (
    OrganizationStruct,
    create_organization,
    delete_organization,
    get_organizations,
)
from ypl.backend.utils.json import json_dumps

router = APIRouter()


@router.post("/organizations", response_model=UUID)
async def create_organization_route(name: str) -> UUID:
    try:
        organization_id = await create_organization(name)
        return organization_id
    except Exception as e:
        log_dict = {
            "message": f"Error creating organization - {str(e)}",
            "name": name,
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/organizations", response_model=list[OrganizationStruct])
async def get_organizations_route(
    name: str | None = Query(None),  # noqa: B008
    exclude_deleted: bool = Query(True),  # noqa: B008
) -> list[OrganizationStruct]:
    try:
        params = locals()
        return await get_organizations(**params)
    except Exception as e:
        log_dict = {
            "message": f"Error getting organizations - {str(e)}",
            "name": name,
            "exclude_deleted": exclude_deleted,
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/organizations/{organization_id}", status_code=204)
async def delete_organization_route(organization_id: UUID) -> None:
    try:
        await delete_organization(organization_id)
    except Exception as e:
        log_dict = {
            "message": f"Error deleting organization - {str(e)}",
            "organization_id": organization_id,
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail=str(e)) from e
