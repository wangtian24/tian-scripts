import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, ORJSONResponse
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from ypl.backend.config import preload_gcp_secrets, settings
from ypl.backend.jobs.app import init_celery, is_not_worker, start_celery_workers, start_redis
from ypl.backend.payment.crypto.crypto_payout import cleanup_crypto_processor, get_processor
from ypl.backend.routes.main import api_router
from ypl.backend.utils.monitoring import start_metrics_manager


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore
    start_redis()
    init_celery()
    workers = start_celery_workers()

    if is_not_worker():
        # only start metrics manager in the main process
        start_metrics_manager()

    # Initialize crypto processor at startup as it is a long running process
    await get_processor()

    # Run preload_gcp_secrets in the background
    asyncio.create_task(preload_gcp_secrets())

    yield  # Hand over to FastAPI.

    await cleanup_crypto_processor()
    for worker in workers:
        worker.terminate()


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=None if settings.ENVIRONMENT == "production" else f"{settings.API_PREFIX}/openapi.json",
    docs_url=None if settings.ENVIRONMENT == "production" else f"{settings.API_PREFIX}/docs",
    redoc_url=None if settings.ENVIRONMENT == "production" else f"{settings.API_PREFIX}/redoc",
    generate_unique_id_function=custom_generate_unique_id,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
    swagger_ui_init_oauth={},
    swagger_ui_parameters={"persistAuthorization": True},
)


def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
        servers=app.servers,
    )

    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    components = openapi_schema["components"]
    components["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-KEY",
            "description": "Regular API key for standard endpoints",
        },
        "AdminApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-ADMIN-API-KEY",
            "description": "Admin API key for administrative endpoints",
        },
    }

    openapi_schema["security"] = [{"ApiKeyAuth": []}, {"AdminApiKeyAuth": []}]

    # Add explicit security requirements for each operation
    for path_item in openapi_schema["paths"].values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                operation["security"] = [{"ApiKeyAuth": []}, {"AdminApiKeyAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Log the error details
    error_details = str(exc.errors())
    logging.warn(f"422 Validation error: {error_details}")

    # Return error response
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "body": exc.body})


# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin).strip("/") for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*", "X-API-KEY", "X-ADMIN-API-KEY"],
        expose_headers=["Content-Disposition", "X-API-KEY", "X-ADMIN-API-KEY"],
    )

app.include_router(api_router, prefix=settings.API_PREFIX)
