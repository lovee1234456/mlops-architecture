import mlflow
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import dagshub

def train_model():
    dagshub.init(repo_owner="loveenair28", repo_name="mlops-architecture", mlflow=True)

    train = pd.read_csv("features/train_features.csv")
    X_train = train.drop(columns=["churn", "customer_id", "event_timestamp"])
    y_train = train["churn"]

    with mlflow.start_run() as run:
        params = {"n_estimators": 200, "max_depth": 8, "random_state": 42}
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, "models/model.pkl")
        run_id = run.info.run_id
        print(f"Training complete. Run ID: {run_id}")
        return {"run_id": run_id, "model_path": "models/model.pkl"}

if __name__ == "__main__":
    train_model()