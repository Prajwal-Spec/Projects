**Salary Prediction Project(Serverless Inference Endpoint):**

**Project Overview:**

This project builds a machine learning pipeline to predict if an individual's salary exceeds $50K/year based on census data. It uses the UCI Adult dataset and implements an XGBoost classifier with preprocessing steps for categorical and numerical features.

**Dataset:**

- Data Source: UCI Adult dataset from the UCI Machine Learning Repository

- Input features include demographics and work-related attributes such as age, workclass, education, occupation, hours-per-week, etc.

- Target variable: Salary >50K (1) or <=50K (0)

**Data Preparation:**

- Raw data downloaded and saved locally, then uploaded to Amazon S3 for SageMaker processing.

- Train/test split maintained from original dataset.

- Categorical features are one-hot encoded; numerical features are standardized.

- Preprocessing combined with the model in a scikit-learn pipeline.

**Model Training on SageMaker:**

- Training script (train.py) uses argparse to accept SageMaker input/output directories and hyperparameters.

- Implements a pipeline with preprocessing + XGBoost classifier.

- Includes accuracy metrics on train and test data printed during training.

- Saves trained model pipeline to the SageMaker model directory.

**Hyperparameter Tuning:**

- SageMaker HyperparameterTuner used to optimize max_depth, learning_rate, and n_estimators.

- Objective metric: Test accuracy extracted from training script logs.

- Parallel tuning jobs run with a max budget of jobs and wait time constraints.

**Model Deployment:**

- Model deployment script (serve.py) defines:

- model_fn for loading model,

- input_fn to parse json input,

- predict_fn to make predictions and return probabilities,

- output_fn to serialize predictions as json.

- Deployment is done using SageMaker SKLearnModel and configured for serverless inference.

- Real-time inference endpoint is created for prediction requests.

- Includes example of sending JSON input for prediction and decoding the response.

**Cleanup:**

- Script includes a helper function to delete the deployed endpoint when no longer needed.

**Dependencies:**

- pandas

- scikit-learn

- xgboost (version 1.7.6)

- fsspec

- s3fs

**How to Run:**

- Prepare data locally and upload to S3 bucket.

- Launch the SageMaker SKLearn estimator with train.py and requirements.txt.

- Tune hyperparameters using SageMaker HyperparameterTuner if desired.

- Deploy the trained model using the serverless inference configuration.

- Send prediction requests to the deployed endpoint using JSON input.

- Clean up endpoint using the provided cleanup function.