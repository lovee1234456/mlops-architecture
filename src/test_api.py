import requests
import pandas as pd

df = pd.read_csv("features/test_features.csv")
row = df.drop(columns=["churn", "customer_id", "event_timestamp"]).iloc[0].to_dict()

response = requests.post("http://127.0.0.1:8000/predict", json=row)
print("Status code:", response.status_code)
print("Response:", response.json())