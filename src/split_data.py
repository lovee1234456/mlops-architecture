import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_data(path="data/raw/customers.csv", target_col="churn"):
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv(path)
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df[target_col])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp[target_col])

    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    print("Split complete:", train.shape, val.shape, test.shape)
    return {"train": "data/processed/train.csv", "val": "data/processed/val.csv", "test": "data/processed/test.csv"}

if __name__ == "__main__":
    split_data()