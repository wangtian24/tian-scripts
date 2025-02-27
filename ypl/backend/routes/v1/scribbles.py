import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ypl.backend.db import get_async_engine
from ypl.backend.utils.json import json_dumps
from ypl.db.scribbles import Scribbles

router = APIRouter()


class ScribbleCreate(BaseModel):
    """Request model for creating a scribble."""

    label: str
    content: dict | None = None


class ScribbleUpdate(BaseModel):
    """Request model for updating a scribble."""

    label: str | None = None
    content: dict | None = None


class ScribbleResponse(BaseModel):
    """Response model for a scribble."""

    scribble_id: uuid.UUID
    label: str
    content: dict | None = None
    created_at: str | None = None
    modified_at: str | None = None


@router.get("/scribbles", response_model=list[ScribbleResponse])
async def get_scribbles(
    limit: Annotated[int, Query(title="Limit", description="Maximum number of scribbles to return", ge=1, le=100)] = 50,
    offset: Annotated[int, Query(title="Offset", description="Number of scribbles to skip", ge=0)] = 0,
) -> list[ScribbleResponse]:
    """
    Get a list of scribbles with pagination.

    Args:
        limit: Maximum number of scribbles to return
        offset: Number of scribbles to skip

    Returns:
        List[ScribbleResponse]: List of scribbles
    """
    try:
        async with AsyncSession(get_async_engine()) as session:
            query = select(Scribbles).order_by(Scribbles.created_at.desc()).offset(offset).limit(limit)  # type: ignore

            result = await session.execute(query)
            scribbles = result.scalars().all()
            session.expunge_all()

            return [
                ScribbleResponse(
                    scribble_id=scribble.scribble_id,
                    label=scribble.label,
                    content=scribble.content,
                    created_at=str(scribble.created_at) if scribble.created_at else None,
                    modified_at=str(scribble.modified_at) if scribble.modified_at else None,
                )
                for scribble in scribbles
            ]

    except Exception as e:
        log_dict = {
            "message": "Failed to fetch scribbles",
            "limit": limit,
            "offset": offset,
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail="Failed to fetch scribbles") from e


@router.get("/scribble", response_model=ScribbleResponse)
async def get_scribble(
    label: Annotated[str, Query(title="Scribble Label", description="The label of the scribble to retrieve")],
) -> ScribbleResponse:
    """
    Get a specific scribble by label.

    Args:
        label: The label of the scribble to retrieve

    Returns:
        ScribbleResponse: The requested scribble

    Raises:
        HTTPException: If the scribble is not found
    """
    try:
        scribble = None
        async with AsyncSession(get_async_engine()) as session:
            query = select(Scribbles).where(Scribbles.label == label)  # type: ignore
            result = await session.execute(query)
            scribble = result.scalar_one_or_none()
            if scribble:
                session.expunge(scribble)

        if not scribble:
            raise HTTPException(status_code=404, detail=f"Scribble with label '{label}' not found")

        return ScribbleResponse(
            scribble_id=scribble.scribble_id,
            label=scribble.label,
            content=scribble.content,
            created_at=str(scribble.created_at) if scribble.created_at else None,
            modified_at=str(scribble.modified_at) if scribble.modified_at else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        log_dict = {
            "message": "Failed to fetch scribble by label",
            "label": label,
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail="Failed to fetch scribble") from e


@router.post("/scribble", response_model=ScribbleResponse)
async def create_scribble(
    scribble: ScribbleCreate,
) -> ScribbleResponse:
    """
    Create a new scribble.

    Args:
        scribble: The scribble data to create

    Returns:
        ScribbleResponse: The created scribble
    """
    try:
        async with AsyncSession(get_async_engine()) as session:
            async with session.begin():
                # Check if a scribble with this label already exists
                query = select(Scribbles).where(Scribbles.label == scribble.label)  # type: ignore
                result = await session.execute(query)
                existing_scribble = result.scalar_one_or_none()

                if existing_scribble:
                    raise HTTPException(
                        status_code=409,
                        detail=f"A scribble with label '{scribble.label}' already exists",
                    )

                # Create the new scribble
                new_scribble = Scribbles(
                    label=scribble.label,
                    content=scribble.content,
                )

                session.add(new_scribble)
                await session.flush()

                # Create a detached copy before closing the session
                return ScribbleResponse(
                    scribble_id=new_scribble.scribble_id,
                    label=new_scribble.label,
                    content=new_scribble.content,
                    created_at=str(new_scribble.created_at) if new_scribble.created_at else None,
                    modified_at=str(new_scribble.modified_at) if new_scribble.modified_at else None,
                )

    except HTTPException:
        raise
    except IntegrityError as e:
        # This catches any integrity errors that might slip through the explicit check
        if "unique constraint" in str(e).lower() and "label" in str(e).lower():
            log_dict = {
                "message": "Duplicate label error",
                "scribble_data": scribble.model_dump(),
                "error": str(e),
            }
            logging.warning(json_dumps(log_dict))
            raise HTTPException(
                status_code=409, detail=f"A scribble with label '{scribble.label}' already exists"
            ) from e
        else:
            log_dict = {
                "message": "Database integrity error",
                "scribble_data": scribble.model_dump(),
                "error": str(e),
            }
            logging.exception(json_dumps(log_dict))
            raise HTTPException(status_code=500, detail="Failed to create scribble") from e
    except Exception as e:
        log_dict = {
            "message": "Failed to create scribble",
            "scribble_data": scribble.model_dump(),
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail="Failed to create scribble") from e


@router.put("/scribble", response_model=ScribbleResponse)
async def update_scribble(
    id: Annotated[uuid.UUID, Query(title="Scribble ID", description="The ID of the scribble to update")],
    scribble: ScribbleUpdate,
) -> ScribbleResponse:
    """
    Update an existing scribble.

    Args:
        id: The UUID of the scribble to update
        scribble: The scribble data to update

    Returns:
        ScribbleResponse: The updated scribble

    Raises:
        HTTPException: If the scribble is not found
    """
    try:
        async with AsyncSession(get_async_engine()) as session:
            async with session.begin():
                stmt = select(Scribbles).where(Scribbles.scribble_id == id)  # type: ignore
                result = await session.execute(stmt)
                existing_scribble = result.scalar_one_or_none()

                if not existing_scribble:
                    raise HTTPException(status_code=404, detail="Scribble not found")

                # Update fields if provided
                if scribble.label is not None:
                    existing_scribble.label = scribble.label
                if scribble.content is not None:
                    existing_scribble.content = scribble.content

                await session.flush()

                # Create a detached copy before closing the session
                return ScribbleResponse(
                    scribble_id=existing_scribble.scribble_id,
                    label=existing_scribble.label,
                    content=existing_scribble.content,
                    created_at=str(existing_scribble.created_at) if existing_scribble.created_at else None,
                    modified_at=str(existing_scribble.modified_at) if existing_scribble.modified_at else None,
                )

    except HTTPException:
        raise
    except Exception as e:
        log_dict = {
            "message": "Failed to update scribble",
            "scribble_id": str(id),
            "scribble_data": scribble.model_dump(),
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail="Failed to update scribble") from e


@router.delete("/scribble", response_model=dict)
async def delete_scribble(
    id: Annotated[uuid.UUID, Query(title="Scribble ID", description="The ID of the scribble to delete")],
) -> dict:
    """
    Delete a scribble.

    Args:
        id: The UUID of the scribble to delete

    Returns:
        dict: A success message

    Raises:
        HTTPException: If the scribble is not found
    """
    try:
        async with AsyncSession(get_async_engine()) as session:
            async with session.begin():
                stmt = select(Scribbles).where(Scribbles.scribble_id == id)  # type: ignore
                result = await session.execute(stmt)
                scribble = result.scalar_one_or_none()

                if not scribble:
                    raise HTTPException(status_code=404, detail="Scribble not found")

                await session.delete(scribble)

                return {"message": "Scribble deleted successfully", "scribble_id": str(id)}

    except HTTPException:
        raise
    except Exception as e:
        log_dict = {
            "message": "Failed to delete scribble",
            "scribble_id": str(id),
            "error": str(e),
        }
        logging.exception(json_dumps(log_dict))
        raise HTTPException(status_code=500, detail="Failed to delete scribble") from e
