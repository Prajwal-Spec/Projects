# Term Deposit Prediction on AWS SageMaker

This project builds, tunes, and deploys an end-to-end Machine Learning pipeline to predict whether a bank client will subscribe to a term deposit. The project leverages **AWS SageMaker** for scalable training, hyperparameter tuning, and deployment using a **Serverless Inference Endpoint**, with a focus on MLOps best practices and cost optimization.

## 📌 Project Overview
The goal is to predict the `y` target variable (Subscription: 0 or 1) using a dataset of client demographics and campaign interactions. The workflow transitions from local data engineering to cloud-based execution on AWS.

**Key Features:**
* **Data Engineering:** Handling data leakage (duration), outlier engineering (pdays), and categorical encoding.
* **Cloud Training:** Custom training script execution on AWS SageMaker instances using Managed Spot Training (saving ~70% cost).
* **Optimization:** Bayesian Hyperparameter Tuning to maximize Model AUC.
* **Deployment:** Serverless Inference Endpoint for auto-scaling real-time predictions.
* **Batch Inference:** Offline processing of the entire dataset using SageMaker Batch Transform.

## 📂 Dataset
The project uses the Bank Marketing Dataset (UCI Machine Learning Repository).

* **Input Features:** Client demographics (age, job, marital), financial indicators (euribor3m, cons.price.idx), and campaign history (pdays, previous).
* **Target:** Binary Class (0: No, 1: Yes).
* **Data Split:** 80% Training, 20% Testing.

## Preprocessing:

* **Leakage Removal:** Dropped duration column.
* **Feature Engineering:** Converted pdays=999 to binary no_previous_contact.
* **Handling Missing:** Median imputation for numericals, 'unknown' for categoricals.

## 🛠️ Tech Stack
* **Python 3.10+**
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `joblib`
* **AWS SDKs:** `boto3`, `sagemaker`
* **Infrastructure:** AWS S3 (Storage), AWS SageMaker (Train/Tuning/Serverless Deploy)

## 🏗️ Model Architecture
The solution uses a custom Scikit-Learn script approach (train.py and serve.py).

* **Preprocessing:** Data is cleaned, One-Hot Encoded, and Scaled (StandardScaler) locally before upload.
* **Classifier:** RandomForestClassifier with class weights balanced to handle the dataset imbalance.

## 🚀 Project Workflow

## 1. Data Preparation & Upload
The raw data is analyzed for anomalies. "999" values in pdays are handled, and leakage features are removed. The processed, scaled datasets (train.csv, test.csv, batch_input.csv) are uploaded to a specific **AWS S3 Bucket.**

## 2. SageMaker Training (train.py)
A custom script src/train.py is generated to handle training on SageMaker instances.

* **Instance:** `ml.m5.large`

* **Spot Instances:** Enabled (Max Wait: 1 hour).

* **Metric:** ROC-AUC and Accuracy.

## 3. Hyperparameter Tuning
A HyperparameterTuner job is run to optimize the Random Forest configuration.

* **Objective:** Maximize Test AUC.

* **Tunable Parameters:**
    * `n_estimators` (50 - 200)
    * `max_depth` (3 - 15)
    * `min_samples_leaf` (2 - 10)

* **Result:** The tuner identified the optimal configuration enabling robust performance on the imbalanced target.

## 4. Deployment (`serve.py`)
The best model is deployed as a Serverless Inference Endpoint.

* **Config:** 2048 MB Memory, Max Concurrency 5.
* **Custom Inference Script:** src/serve.py handles JSON Lines input parsing and returns both the prediction class and the probability score.

## 5. Batch Transform
A Batch Transform job is executed on the full dataset (~41k rows) to validate global model performance without keeping a real-time endpoint active.

* **Final Accuracy:** ~90.5%
* **Final AUC:** ~0.80

## 💻 Usage

## Prerequisites
Ensure you have an AWS account with SageMaker permissions and the necessary Python libraries installed.

pip install -r src/requirements.txt