import pandas as pd
from sklearn.model_selection import train_test_split

def sample_data(
    input_path="data/raw/customer_churn_1m.csv",
    output_path="data/raw/customers.csv",
    n_samples=250_000,
    target_col="churn"
):
    df = pd.read_csv(input_path)
    print(f"Full dataset: {df.shape}")

    # Stratified sample using train_test_split — keeps churn ratio intact, keeps all columns
    sampled, _ = train_test_split(
        df,
        train_size=n_samples,
        stratify=df[target_col],
        random_state=42
    )

    sampled.to_csv(output_path, index=False)
    print(f"Sampled dataset: {sampled.shape}")
    print(f"Churn ratio (full):    {df[target_col].mean():.3f}")
    print(f"Churn ratio (sampled): {sampled[target_col].mean():.3f}")

if __name__ == "__main__":
    sample_data()