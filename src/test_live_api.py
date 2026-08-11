import requests
import pandas as pd

LIVE_URL = "https://churn-prediction-api-6oo8.onrender.com"  # use your exact URL from Render's dashboard

df = pd.read_csv("features/test_features.csv")
row = df.drop(columns=["churn", "customer_id", "event_timestamp"]).iloc[0].to_dict()

response = requests.post(f"{LIVE_URL}/predict", json=row)
print("Status code:", response.status_code)
print("Response:", response.json())