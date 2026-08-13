import pandas as pd
import os

CATEGORICAL_COLS = ["gender", "education", "marital_status", "contract", "payment_method", "paperless_billing"]
NUMERIC_COLS_TO_IMPUTE = ["annual_income", "customer_satisfaction", "num_complaints", "avg_monthly_gb", "credit_score"]

def _encode(df, medians=None):
    df = df.copy()

    # Keep customer_id — Feast needs it as the lookup key (entity)
    # Keep signup_date as event_timestamp — Feast needs this to track when features were recorded
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df = df.rename(columns={"signup_date": "event_timestamp"})

    df["signup_year"] = df["event_timestamp"].dt.year
    df["signup_month"] = df["event_timestamp"].dt.month

    # Fill missing numeric values with the median (computed from TRAINING data only, to avoid leakage)
    for col in NUMERIC_COLS_TO_IMPUTE:
        if col in df.columns:
            df[col] = df[col].fillna(medians[col])

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    return df

def run_feature_engineering():
    # Compute medians from the TRAINING set only — using val/test data to fill values
    # would leak information from data the model shouldn't "see" during training
    os.makedirs("features", exist_ok=True)
    train_raw = pd.read_csv("data/processed/train.csv")
    medians = {col: train_raw[col].median() for col in NUMERIC_COLS_TO_IMPUTE}
    print("Medians used for imputation (from train set):", medians)

    paths = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"data/processed/{split}.csv")
        feat_df = _encode(df, medians=medians)
        out_path = f"features/{split}_features.csv"
        feat_df.to_csv(out_path, index=False)
        paths[split] = out_path
        print(f"{split}: {feat_df.shape}, missing values remaining: {feat_df.isnull().sum().sum()}")
    return paths

if __name__ == "__main__":
    run_feature_engineering()