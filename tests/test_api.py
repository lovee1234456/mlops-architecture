import requests
import pandas as pd

BASE_URL = "http://127.0.0.1:8000"

def load_sample_customers(n=5):
    df = pd.read_csv("features/test_features.csv")
    return df.drop(columns=["churn", "customer_id", "event_timestamp"]).head(n).to_dict(orient="records")

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("PASS: /health check")

def test_predict_multiple_real_customers():
    customers = load_sample_customers(5)
    for i, customer in enumerate(customers):
        response = requests.post(f"{BASE_URL}/predict", json=customer)
        assert response.status_code == 200, f"Customer {i} failed with status {response.status_code}"
        body = response.json()
        assert "churn_probability" in body
        assert "churn_prediction" in body
        assert 0.0 <= body["churn_probability"] <= 1.0, f"Probability out of range: {body['churn_probability']}"
        print(f"PASS: customer {i} -> probability={body['churn_probability']}, prediction={body['churn_prediction']}")

def test_predict_missing_fields():
    # Send an incomplete request - only 2 fields instead of the full 40
    response = requests.post(f"{BASE_URL}/predict", json={"age": 30, "tenure": 5})
    assert response.status_code == 200, "API should handle missing fields gracefully, not crash"
    body = response.json()
    assert "churn_probability" in body
    print(f"PASS: missing fields handled -> {body}")

def test_predict_empty_request():
    response = requests.post(f"{BASE_URL}/predict", json={})
    assert response.status_code == 200, "API should handle a fully empty request without crashing"
    print(f"PASS: empty request handled -> {response.json()}")

if __name__ == "__main__":
    test_health()
    test_predict_multiple_real_customers()
    test_predict_missing_fields()
    test_predict_empty_request()
    print("\nAll tests passed.")