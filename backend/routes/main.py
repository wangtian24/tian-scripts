from fastapi import APIRouter

from backend.routes.v1 import health

api_router = APIRouter()
api_router.include_router(health.router, prefix="/v1", tags=["health"])
