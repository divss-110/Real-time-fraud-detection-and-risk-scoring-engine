import os

os.environ.setdefault("FEATURE_BACKEND", "local")

import pytest
from fastapi.testclient import TestClient

from src.config import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="Train the model first")

TXN = {
    "transaction_id": "test_1",
    "user_id": 7,
    "amount": 950.0,
    "hour": 3,
    "merchant_category": 7,
    "is_foreign": 1,
    "card_present": 0,
    "device_change": 1,
    "distance_from_home_km": 800.0,
    "seconds_since_last_txn": 30.0,
}


@pytest.fixture(scope="module")
def client():
    from src.inference.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_score_returns_shap_and_is_fast(client):
    body = client.post("/score", json=TXN).json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert len(body["shap_values"]) == 17
    assert body["latency_ms"]["total"] < 50


def test_invalid_input_rejected(client):
    assert client.post("/score", json={**TXN, "hour": 30}).status_code == 422
