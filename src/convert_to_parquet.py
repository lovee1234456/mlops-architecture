import pandas as pd

df = pd.read_csv("features/train_features.csv")
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
df.to_parquet("feature_store/churn_repo/feature_repo/data/train_features.parquet")
print(f"Saved parquet: {df.shape}")