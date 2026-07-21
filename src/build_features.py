import pandas as pd

CATEGORICAL_COLS = ["gender", "education", "marital_status", "contract", "payment_method", "paperless_billing"]

def _encode(df):
    df = df.copy()

    # Drop identifier - not a real feature
    df = df.drop(columns=["customer_id"])

    # Turn signup_date into numeric features a model can actually use
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["signup_year"] = df["signup_date"].dt.year
    df["signup_month"] = df["signup_date"].dt.month
    df = df.drop(columns=["signup_date"])

    # One-hot encode categorical columns
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
    