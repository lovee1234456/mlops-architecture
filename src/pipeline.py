from src.data_validation import validate
from src.split_data import split_data
from src.build_features import run_feature_engineering
from src.train import train_model
from src.evaluate import evaluate_model

def run_pipeline():
    print("=== Step 1: Validating raw data ===")
    validate()

    print("\n=== Step 2: Splitting data ===")
    split_data()

    print("\n=== Step 3: Building features ===")
    run_feature_engineering()

    print("\n=== Step 4: Training model ===")
    model_info = train_model()

    print("\n=== Step 5: Evaluating model ===")
    metrics = evaluate_model(model_info["model_path"])

    print("\n=== Pipeline completed successfully ===")
    print(f"Final F1: {metrics['f1']:.4f}")
    print(f"Model run ID: {model_info['run_id']}")

if __name__ == "__main__":
    run_pipeline()