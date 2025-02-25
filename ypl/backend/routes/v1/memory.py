import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ypl.backend.llm.memories import get_memories
from ypl.backend.utils.json import json_dumps

router = APIRouter()


class MemoryEntry(BaseModel):
    memory_id: UUID
    user_id: str
    memory_content: str
    source_message_id: UUID


class MemoriesResponse(BaseModel):
    memories: list[MemoryEntry]
    has_more_rows: bool


@router.get("/list_memories", response_model=MemoriesResponse)
async def list_memories_route(
    user_id: str | None = Query(None),  # noqa: B008
    message_id: UUID | None = Query(None),  # noqa: B008
    chat_id: UUID | None = Query(None),  # noqa: B008
    limit: int = Query(50),  # noqa: B008
    offset: int = Query(0),  # noqa: B008
) -> MemoriesResponse:
    params = locals()
    try:
        memories, has_more_rows = await get_memories(**params)
        entries = [MemoryEntry(**memory.model_dump(include=MemoryEntry.model_fields)) for memory in memories]
        return MemoriesResponse(memories=entries, has_more_rows=has_more_rows)
    except Exception as e:
        logging.exception(
            json_dumps(
                {
                    "message": f"Error getting memories: {e}",
                    "params": params,
                }
            )
        )
        raise HTTPException(status_code=500, detail=str(e)) from e
