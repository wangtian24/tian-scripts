import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, ORJSONResponse
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from ypl.backend.config import preload_gcp_secrets, settings
from ypl.backend.jobs.app import init_celery, start_celery_workers, start_redis
from ypl.backend.payment.crypto.crypto_payout import cleanup_crypto_processor, get_processor
from ypl.backend.routes.main import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore
    start_redis()
    init_celery()
    workers = start_celery_workers()
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
)


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
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

app.include_router(api_router, prefix=settings.API_PREFIX)
