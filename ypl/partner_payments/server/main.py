from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.routing import APIRoute

from ypl.partner_payments.server.config import settings
from ypl.partner_payments.server.partner.clients import partner_clients
from ypl.partner_payments.server.routes.all import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore
    await partner_clients.initialize()
    yield  # Hand over to FastAPI.
    await partner_clients.cleanup()


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


app = FastAPI(
    title=f"{settings.PROJECT_NAME}",
    openapi_url=None if settings.ENVIRONMENT == "production" else f"{settings.API_PREFIX}/openapi.json",
    docs_url=None if settings.ENVIRONMENT == "production" else f"{settings.API_PREFIX}/docs",
    redoc_url=None if settings.ENVIRONMENT == "production" else f"{settings.API_PREFIX}/redoc",
    generate_unique_id_function=custom_generate_unique_id,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

app.include_router(api_router)
