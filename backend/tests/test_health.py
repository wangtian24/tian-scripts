from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routes.v1.health import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
