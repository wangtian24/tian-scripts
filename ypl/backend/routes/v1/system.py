import logging

from fastapi import APIRouter, Depends, Header, HTTPException

from ypl.backend.llm.db_helpers import (
    deduce_original_provider,
    deduce_original_providers,
    deduce_semantic_groups,
    get_active_models,
    get_all_fast_models,
    get_all_live_models,
    get_all_pro_and_strong_models,
    get_all_pro_models,
    get_all_reasoning_models,
    get_all_strong_models,
    get_image_attachment_models,
    get_model_context_lengths,
    get_model_creation_dates,
    get_pdf_attachment_models,
    get_yapp_descriptions,
)
from ypl.backend.llm.provider.provider_clients import load_models_with_providers
from ypl.backend.llm.routing.rule_router import get_routing_table
from ypl.backend.utils.json import json_dumps
from ypl.backend.utils.soul_utils import SoulPermission, validate_permissions

router = APIRouter()

# Maps cache names that can be cleared to list of cached functions to clear.
CACHED_FUNCS: dict[str, list] = {
    "routing-table": [
        get_routing_table,
        deduce_original_provider,
        deduce_original_providers,
    ],
    "active-models": [
        get_active_models,
        get_image_attachment_models,
        get_pdf_attachment_models,
        load_models_with_providers,
        deduce_original_provider,
        deduce_original_providers,
        deduce_semantic_groups,
        get_all_pro_models,
        get_all_strong_models,
        get_all_pro_and_strong_models,
        get_all_fast_models,
        get_all_live_models,
        get_all_reasoning_models,
        get_model_creation_dates,
        get_model_context_lengths,
        get_yapp_descriptions,
    ],
}


async def validate_manage_caches(
    x_creator_email: str | None = Header(None, alias="X-Creator-Email"),
) -> None:
    """Validate that the user has MANAGE_CACHES permission."""
    await validate_permissions([SoulPermission.MANAGE_CACHES], x_creator_email)


@router.post("/admin/clear-cache", dependencies=[Depends(validate_manage_caches)])
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
        cache_infos[func.__name__] = str(func.cache_info())
        func.cache_clear()
    logging.info(json_dumps({"message": f"Cleared {name} cache", "info": cache_infos}))
    return cache_infos
