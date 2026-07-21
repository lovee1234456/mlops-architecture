import pandas as pd

CATEGORICAL_COLS = ["gender", "education", "marital_status", "contract", "payment_method", "paperless_billing"]

def _encode(df):
    df = df.copy()

    # Keep customer_id — Feast needs it as the lookup key (entity)
    # Keep signup_date as event_timestamp — Feast needs this to track when features were recorded
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df = df.rename(columns={"signup_date": "event_timestamp"})

    df["signup_year"] = df["event_timestamp"].dt.year
    df["signup_month"] = df["event_timestamp"].dt.month

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    return df

def run_feature_engineering():
    paths = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"data/processed/{split}.csv")
        feat_df = _encode(df)
        out_path = f"features/{split}_features.csv"
        feat_df.to_csv(out_path, index=False)
        paths[split] = out_path
        print(f"{split}: {feat_df.shape}")
    return paths

if __name__ == "__main__":
    run_feature_engineering()