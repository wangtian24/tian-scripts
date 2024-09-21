from collections.abc import Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from ypl.backend.config import settings
from ypl.backend.routes.main import api_router


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    generate_unique_id_function=custom_generate_unique_id,
    default_response_class=ORJSONResponse,
)


@app.middleware("http")
async def origin_header_check(request: Request, call_next: Callable) -> Response:
    origin = request.headers.get("origin")
    if origin is None:
        raise HTTPException(status_code=400, detail="Origin header is required")
    response: Response = await call_next(request)
    return response


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
