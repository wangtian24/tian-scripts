"""Tests for the Coinbase webhook handler."""
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ypl.webhooks.server import app

client = TestClient(app)


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    return {
        "type": "charge:created",
        "id": "123",
        "data": {"code": "ABC123", "name": "Test Charge", "amount": "100.00", "currency": "USD"},
    }


def test_webhook_requires_signature() -> None:
    """Test that webhook requires X-Coinbase-Signature header."""
    response = client.post("/coinbase/webhook/test-token", json={})
    assert response.status_code == 400
    assert response.json()["detail"] == "Bad request"


def test_webhook_requires_valid_json() -> None:
    """Test that webhook requires valid JSON payload."""
    response = client.post(
        "/coinbase/webhook/test-token", headers={"X-Coinbase-Signature": "test-signature"}, content="invalid json"
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Bad request"


def test_webhook_success(valid_payload: dict[str, Any]) -> None:
    """Test successful webhook request."""
    response = client.post(
        "/coinbase/webhook/test-token", headers={"X-Coinbase-Signature": "test-signature"}, json=valid_payload
    )
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
