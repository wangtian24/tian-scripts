from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ypl.backend.routes.v1.health import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)


@patch("ypl.backend.routes.v1.health.AsyncSession")
def test_health(mock_session: AsyncMock) -> None:
    # Mock the database session and query result
    mock_session_instance = AsyncMock()
    mock_session.return_value.__aenter__.return_value = mock_session_instance

    mock_result = AsyncMock()
    mock_result.scalar = lambda: "abc123"
    mock_session_instance.exec.return_value = mock_result

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db_version": "abc123"}
