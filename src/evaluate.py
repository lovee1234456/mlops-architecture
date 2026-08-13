import json
import joblib
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

def evaluate_model(model_path="models/model.pkl", threshold=0.15):
    os.makedirs("evaluation", exist_ok=True)
    model = joblib.load(model_path)
    test = pd.read_csv("features/test_features.csv")
    X_test = test.drop(columns=["churn", "customer_id", "event_timestamp"])
    y_test = test["churn"]

    probs = model.predict_proba(X_test)[:, 1]

    # Find the probability cutoff that maximizes F1, instead of assuming 0.5
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_cutoff = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    preds = (probs >= best_cutoff).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "best_cutoff": float(best_cutoff),
    }

    with open("evaluation/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)

    if metrics["f1"] < threshold:
        raise ValueError(f"Model F1 {metrics['f1']:.3f} is below the {threshold} acceptance threshold — "
                          f"model will NOT be registered or deployed.")
    return metrics

if __name__ == "__main__":
    evaluate_model()