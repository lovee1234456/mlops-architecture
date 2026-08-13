import pandas as pd
import json
from scipy.stats import ks_2samp

NUMERIC_FEATURES = [
    "age", "annual_income", "dependents", "tenure", "monthlycharges",
    "totalcharges", "num_services", "customer_satisfaction", "num_complaints",
    "num_service_calls", "late_payments", "avg_monthly_gb",
    "days_since_last_interaction", "credit_score",
]

def check_drift(reference_path="features/train_features.csv", current_path=None, alpha=0.05):
    """
    Compares each numeric feature's distribution between reference (training) data
    and current (recent/live) data using the Kolmogorov-Smirnov test.

    A low p-value (< alpha) means the two distributions are significantly different -
    i.e., the feature has drifted.
    """
    reference = pd.read_csv(reference_path)

    if current_path:
        current = pd.read_csv(current_path)
    else:
        raise ValueError("Must provide current_path pointing to recent data to compare against")

    results = {}
    drifted_features = []

    for feature in NUMERIC_FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            continue

        stat, p_value = ks_2samp(reference[feature].dropna(), current[feature].dropna())
        drifted = p_value < alpha

        results[feature] = {
            "ks_statistic": float(stat),
            "p_value": float(p_value),
            "drifted": bool(drifted),
        }
        if drifted:
            drifted_features.append(feature)

    summary = {
        "total_features_checked": len(results),
        "drifted_feature_count": len(drifted_features),
        "drifted_features": drifted_features,
        "overall_drift_detected": len(drifted_features) > 0,
        "details": results,
    }

    with open("monitoring/logs/drift_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Checked {summary['total_features_checked']} features.")
    print(f"Drifted: {summary['drifted_feature_count']} -> {drifted_features}")
    print(f"Report saved to monitoring/logs/drift_report.json")

    return summary

if __name__ == "__main__":
    # For now, compare train vs test as a sanity check (should show LOW drift, since both came from the same source)
    check_drift(reference_path="features/train_features.csv", current_path="features/test_features.csv")