from fastapi import FastAPI
import joblib
import json
import pandas as pd
import datetime

app = FastAPI(title="Churn Prediction API")

model = joblib.load("models/model.pkl")

with open("models/feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

with open("evaluation/metrics.json") as f:
    METRICS = json.load(f)
    BEST_CUTOFF = METRICS["best_cutoff"]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: dict):
    # Build a single-row DataFrame, filling any missing expected columns with 0
    row = {col: features.get(col, 0) for col in FEATURE_COLUMNS}
    df = pd.DataFrame([row])[FEATURE_COLUMNS]  # enforce exact column order

    prob = model.predict_proba(df)[0][1]
    prediction = bool(prob >= BEST_CUTOFF)

    log_entry = {
        "timestamp": str(datetime.datetime.utcnow()),
        "input": features,
        "churn_probability": float(prob),
        "prediction": prediction,
    }
    with open("monitoring/logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"churn_probability": round(float(prob), 4), "churn_prediction": prediction}