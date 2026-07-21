import pandas as pd

def validate(path="data/raw/customers.csv", target_col="churn"):
    df = pd.read_csv(path)
    assert df.isnull().sum().sum() < len(df) * 0.3, "Too many missing values"
    assert target_col in df.columns, f"Target column '{target_col}' missing"
    print(f"Validation passed: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    validate()