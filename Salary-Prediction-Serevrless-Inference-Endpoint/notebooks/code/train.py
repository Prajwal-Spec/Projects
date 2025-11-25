
import argparse
import os
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model_file_name = "pipeline_model1.joblib"

# Main function
def main():
    # Arguments
    parser = argparse.ArgumentParser()

    # Inbuilt Arguments
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    # Add arguments for data directories
    # SageMaker passes these automatically if you use inputs={...} in the estimator
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # Hyperparameters to Tune
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--n_estimators", type=int, default=100)
    
    # Custom Arguements
    parser.add_argument("--use_label_encoder", default=False)
    parser.add_argument("--eval_metric", type=str, default="logloss")

    args, _ = parser.parse_known_args()

    # Load data
    # Read from local container paths, not S3
    # We join the channel path with the filename
    print(f"Reading training data from: {args.train}")
    train_df = pd.read_csv(os.path.join(args.train, "train.csv"))
    
    print(f"Reading test data from: {args.test}")
    test_df = pd.read_csv(os.path.join(args.test, "test.csv"))

   # Split features and targets
    X_train = train_df.drop("target", axis=1)
    y_train = train_df['target']

    X_test = test_df.drop("target", axis=1)
    y_test = test_df['target']
    
    # Define columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64"]).columns.tolist()
    
    # One Hot Encode the catgorial columns
    cat_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

    # Scale the numerical feature
    num_transformer = StandardScaler()

    # Combine preprocessing in a ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    # Ml model
    xgb = XGBClassifier(
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        use_label_encoder=False,
        eval_metric='logloss')

    # Pipeline with preprocessor + Ml model
    pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("xgb_model",xgb )
    ])

    # Fit the pipeline locally
    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    
    # Compute accuracy on training data 
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"Train Accuracy: {train_acc:.4f}")

    # Compute accuracy on test data
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the model
    model_save_path = os.path.join(args.model_dir, model_file_name)
    joblib.dump(pipeline, model_save_path)
    print(f"Model saved at {model_save_path}")

# Run the main function when the script runs
if __name__ == "__main__":
    main()
