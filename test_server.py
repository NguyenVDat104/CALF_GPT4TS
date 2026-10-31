import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_health_check():
    """Test standard health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "CALF" in data["framework"]

def test_list_datasets():
    """Test dataset listing API."""
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert isinstance(data["datasets"], list)

def test_forecast_inference_default():
    """Test forecast inference on default settings."""
    payload = {
        "dataset": "BTCUSDT_15m_calf.csv",
        "variable": "close",
        "seq_len": 96,
        "pred_len": 96,
        "backbone": "dinov3",
        "encoder": "pca",
        "stride": 1,
        "distill_loss": "cosine",
        "feature_w": 0.01
    }
    response = client.post("/api/forecast", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert "data" in result
    assert "metrics" in result["data"]
    assert "mae" in result["data"]["metrics"]
    assert "rmse" in result["data"]["metrics"]
