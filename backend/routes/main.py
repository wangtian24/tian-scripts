from fastapi import APIRouter

from backend.routes.v1 import health, highlight_similar_content
from backend.routes.v1 import route as llm_route

api_router = APIRouter()
api_router.include_router(health.router, prefix="/v1", tags=["health"])
api_router.include_router(highlight_similar_content.router, prefix="/v1", tags=["highlight"])
api_router.include_router(llm_route.router, prefix="/v1", tags=["route"])
