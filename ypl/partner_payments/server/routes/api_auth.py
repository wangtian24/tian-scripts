import os

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from ypl.backend.config import settings

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False, description="Regular API key for standard endpoints")


async def validate_api_key(api_key_header: str = Security(api_key_header)) -> bool:
    environment = os.getenv("ENVIRONMENT")

    if environment == "local":
        return True
    if api_key_header == settings.X_API_KEY:
        return True
    else:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Could not validate API KEY")
