import mlflow
import mlflow.xgboost
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import dagshub

def train_model():
    dagshub.init(repo_owner="loveenair28", repo_name="mlops-architecture", mlflow=True)

    train = pd.read_csv("features/train_features.csv")
    X_train = train.drop(columns=["churn", "customer_id", "event_timestamp"])
    y_train = train["churn"]

    # scale_pos_weight tells XGBoost how imbalanced the classes are (majority/minority ratio)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
    }

    with mlflow.start_run() as run:
        search = GridSearchCV(
            XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric="logloss"),
            param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            verbose=2,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("cv_best_f1", search.best_score_)

        for i, (params, mean_score) in enumerate(zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])):
            print(f"Trial {i}: {params} -> F1 = {mean_score:.4f}")

        model_info = mlflow.xgboost.log_model(best_model, "model")
        joblib.dump(best_model, "models/model.pkl")

        run_id = run.info.run_id

        # Register this run's model in the MLflow Model Registry, using the confirmed URI
        result = mlflow.register_model(
            model_uri=model_info.model_uri,
            name="churn-classifier"
        )
        print(f"Registered as version {result.version} of 'churn-classifier'")
        print(f"\nBest params: {search.best_params_}")
        print(f"Best CV F1: {search.best_score_:.4f}")
        print(f"Training complete. Run ID: {run_id}")
        return {"run_id": run_id, "model_path": "models/model.pkl"}

if __name__ == "__main__":
    train_model()