
import argparse
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model_file_name = "pipeline_modelA.joblib"

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
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--min_samples_split", type=float, default=0.05)
    parser.add_argument("--criterion", type=str, default="gini")
    
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
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Scale the numerical features
    sc = StandardScaler()

    # Column transformer to apply transformation on numerical columns
    ct = ColumnTransformer([
    ("Scaling", sc, num_cols)
    ])

    # Random Forest Model
    rfc = RandomForestClassifier(n_estimators=args.n_estimators, 
                                 min_samples_split=args.min_samples_split,
                                 criterion=args.criterion)

    # Sklearn pipeline to combine feature engineering and ML model
    pipeline_rfc_model = Pipeline([
    ("Data Transformations", ct),
    ("Random Forest Model", rfc)
    ])
    
    # Fit the model locally
    pipeline_rfc_model.fit(X_train, y_train)
    
    y_pred_train = pipeline_rfc_model.predict(X_train)
    y_pred_test = pipeline_rfc_model.predict(X_test)
    
    # Compute accuracy on training data 
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"Train Accuracy: {train_acc:.4f}")

    # Compute accuracy on test data
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the model
    model_save_path = os.path.join(args.model_dir, model_file_name)
    joblib.dump(pipeline_rfc_model, model_save_path)
    print(f"Model saved at {model_save_path}")

# Run the main function when the script runs
if __name__ == "__main__":
    main()
