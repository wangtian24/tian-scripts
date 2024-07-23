from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    # TODO(gm): actually check that the server is up.
    return {"status": "ok"}
