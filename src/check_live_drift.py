import json
import pandas as pd
from src.drift_check import check_drift

def load_prediction_logs(path="monitoring/logs/predictions.jsonl"):
    records = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            record = entry["input"]
            # Skip obviously invalid/test records - e.g. tenure of 0 combined with age of 0
            # is not a real customer, just an empty/malformed test request
            if record.get("age", 0) == 0 and record.get("tenure", 0) == 0:
                continue
            records.append(record)
    return pd.DataFrame(records)

if __name__ == "__main__":
    live_data = load_prediction_logs()
    live_data.to_csv("monitoring/logs/live_features_snapshot.csv", index=False)
    print(f"Loaded {len(live_data)} real (non-empty) prediction records")

    if len(live_data) < 100:
        print("Too few real records for a meaningful drift check yet - need more real traffic.")
    else:
        check_drift(reference_path="features/train_features.csv", current_path="monitoring/logs/live_features_snapshot.csv")