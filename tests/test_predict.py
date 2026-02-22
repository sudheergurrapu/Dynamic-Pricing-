from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_predict_endpoint_returns_expected_fields():
    payload = {
        "base_price": 120,
        "demand_index": 0.95,
        "competitor_price": 110,
        "inventory_level": 180,
        "customer_segment": "regular",
        "month": 10,
        "day_of_week": 3,
        "historical_demand": [0.8, 0.85, 0.9, 0.92, 0.95],
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    for key in [
        "optimized_price",
        "bounded_price",
        "revenue_prediction",
        "confidence_score",
        "selected_model",
        "model_metrics",
        "kpis",
    ]:
        assert key in body

    assert payload["base_price"] * 0.7 <= body["bounded_price"] <= payload["base_price"] * 1.5
