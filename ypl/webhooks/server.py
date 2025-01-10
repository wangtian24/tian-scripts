import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, ORJSONResponse
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from ypl.backend.config import settings
from ypl.webhooks.routes.main import api_router


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


app = FastAPI(
    title=f"{settings.PROJECT_NAME} Webhooks",
    openapi_url=None if settings.ENVIRONMENT == "production" else f"{settings.API_PREFIX}/webhooks/openapi.json",
    docs_url=None if settings.ENVIRONMENT == "production" else f"{settings.API_PREFIX}/webhooks/docs",
    redoc_url=None if settings.ENVIRONMENT == "production" else f"{settings.API_PREFIX}/webhooks/redoc",
    generate_unique_id_function=custom_generate_unique_id,
    default_response_class=ORJSONResponse,
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


# Include the main API router with partner-specific routes
app.include_router(api_router)
