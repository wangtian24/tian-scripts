import logging

from dotenv import load_dotenv
from fastapi import APIRouter, Depends

from ypl.backend.routes.api_auth import validate_api_key
from ypl.pytorch.serve.routes.v1 import category, health

logger = logging.getLogger(__name__)


api_router = APIRouter(dependencies=[Depends(validate_api_key)])
api_router.include_router(health.router, prefix="/v1", tags=["health"])
api_router.include_router(category.router, prefix="/v1", tags=["categorize"])

load_dotenv()
