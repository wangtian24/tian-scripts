import logging
import os

from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

from ypl.backend.config import settings

load_dotenv()

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False, description="Regular API key for standard endpoints")
api_key = os.getenv("X_API_KEY")

admin_api_key_header = APIKeyHeader(
    name="X-ADMIN-API-KEY", auto_error=False, description="Admin API key for admin endpoints"
)


async def validate_api_key(api_key_header: str = Security(api_key_header)) -> bool:
    environment = os.getenv("ENVIRONMENT")

    if api_key_header is None:
        log_dict = {"message": "validate_api_key: api key header is None", "environment": environment}
        logging.info(log_dict)

    if environment == "local":
        return True
    if api_key_header == api_key:
        return True
    else:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Could not validate API KEY")


async def validate_admin_api_key(api_key_header: str = Security(admin_api_key_header)) -> bool:
    if api_key_header is None:
        log_dict = {"message": "validate_admin_api_key: api key header is None", "environment": settings.ENVIRONMENT}
        logging.info(log_dict)

    if settings.ENVIRONMENT == "local":
        return True
    if api_key_header == settings.X_ADMIN_API_KEY:
        return True
    else:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Could not validate Admin API KEY")
