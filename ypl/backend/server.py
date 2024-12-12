from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from ypl.backend.config import settings
from ypl.backend.jobs.app import init_celery, start_celery_workers, start_redis
from ypl.backend.routes.main import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore
    start_redis()
    init_celery()
    workers = start_celery_workers()

    yield  # Hand over to FastAPI.

    for worker in workers:
        worker.terminate()


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    generate_unique_id_function=custom_generate_unique_id,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


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
