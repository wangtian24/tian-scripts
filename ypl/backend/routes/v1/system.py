import logging

from fastapi import APIRouter, HTTPException

from ypl.backend.llm.db_helpers import get_active_models, get_image_attachment_models, get_pdf_attachment_models
from ypl.backend.llm.provider.provider_clients import load_active_models_with_providers
from ypl.backend.llm.routing.rule_router import get_routing_table
from ypl.backend.utils.json import json_dumps

router = APIRouter()

# Maps cache names that can be cleared to list of cached functions to clear.
CACHED_FUNCS: dict[str, list] = {
    "routing-table": [get_routing_table],
    "active-models": [
        get_active_models,
        get_image_attachment_models,
        get_pdf_attachment_models,
        load_active_models_with_providers,
    ],
}


@router.post("/admin/clear-cache")
async def clear_cache(name: str) -> dict[str, str]:
    """Clears a cache and returns its pre-clear info."""
    funcs = CACHED_FUNCS.get(name)
    if not funcs:
        raise HTTPException(status_code=400, detail=f"Unknown cache: {name}")

    cache_infos = {}
    for func in funcs:
        if not hasattr(func, "cache_clear") or not hasattr(func, "cache_info"):
            raise HTTPException(
                status_code=400,
                detail=f"Function {func.__name__} (from {name} cache) does not support cache_clear/cache_info",
            )
        cache_infos[func.__name__] = func.cache_info()
        func.cache_clear()
    logging.info(json_dumps({"message": f"Cleared {name} cache", "info": cache_infos}))
    return cache_infos
