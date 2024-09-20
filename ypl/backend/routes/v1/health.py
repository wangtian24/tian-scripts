from fastapi import APIRouter, Depends

from ..api_auth import validate_api_key

router = APIRouter(dependencies=[Depends(validate_api_key)])


@router.get("/health")
def health() -> dict[str, str]:
    # TODO(gm): actually check that the server is up.
    return {"status": "ok"}
