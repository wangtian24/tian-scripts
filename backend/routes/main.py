from fastapi import APIRouter

from backend.routes.v1 import health, highlight_similar_content

api_router = APIRouter()
api_router.include_router(health.router, prefix="/v1", tags=["health"])
api_router.include_router(highlight_similar_content.router, prefix="/v1", tags=["group_contents"])
