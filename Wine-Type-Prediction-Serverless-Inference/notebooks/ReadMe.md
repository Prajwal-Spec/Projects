# Wine Type Prediction on AWS SageMaker

This project builds, tunes, and deploys a Machine Learning pipeline to classify wine varieties based on their chemical properties. The project leverages **AWS SageMaker** for scalable training, hyperparameter tuning, and deployment using a **Serverless Inference Endpoint**.

## 📌 Project Overview
The goal is to predict the `target` variable (Wine Class: 0, 1, or 2) using a dataset of chemical analysis features. The workflow transitions from local testing to cloud-based execution on AWS.

**Key Features:**
* **EDA:** Automated profiling using `ydata-profiling`.
* **Pipeline:** Scikit-learn `Pipeline` combining data preprocessing (scaling) and model training.
* **Cloud Training:** Custom training script execution on AWS SageMaker instances (`ml.m5.large`).
* **Optimization:** Bayesian Hyperparameter Tuning to find the best model configuration.
* **Deployment:** Serverless Inference Endpoint for cost-effective real-time predictions.

## 📂 Dataset
The project uses the [UCI Wine Dataset](https://archive.ics.uci.edu/ml/datasets/Wine).

* **Input Features:** 13 continuous chemical properties (Alcohol, Malic Acid, Ash, Magnesium, Flavanoids, Color Intensity, Hue, Proline, etc.).
* **Target:** 3 Classes (mapped from original 1, 2, 3 to **0, 1, 2**).
* **Data Split:** 90% Training, 10% Testing.

## 🛠️ Tech Stack
* **Python 3.12**
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `joblib`, `ydata-profiling`
* **AWS SDKs:** `boto3`, `sagemaker`
* **Infrastructure:** AWS S3 (Storage), AWS SageMaker (Train/Deploy)

## 🏗️ Model Architecture
The solution uses a scikit-learn `Pipeline` to ensure preprocessing steps are saved with the model artifact.

1.  **Preprocessing:** `StandardScaler` applied to all numerical features via `ColumnTransformer`.
2.  **Classifier:** `RandomForestClassifier`.

## 🚀 Project Workflow

### 1. Data Preparation & Upload
The data is loaded, cleaned, and split. The target variable is binarized (0-indexed). The training and testing CSVs are uploaded to a default **AWS S3 Bucket**.

### 2. Local Training (Sanity Check)
Before moving to the cloud, the pipeline is tested locally within the notebook to ensure the `ColumnTransformer` and `RandomForest` integration works as expected.
* *Local Accuracy:* ~100%

### 3. SageMaker Training (`train.py`)
A custom script `train.py` is generated to handle training on SageMaker instances.
* **Instance:** `ml.m5.large`
* **Spot Instances:** Enabled (for cost savings).
* **Framework:** Scikit-learn 1.2-1.

### 4. Hyperparameter Tuning
A `HyperparameterTuner` job is run to optimize the model.
* **Metric:** Test Accuracy.
* **Tunable Parameters:**
    * `n_estimators` (1 - 20)
    * `min_samples_split` (0.01 - 0.5)
    * `criterion` (gini vs. entropy)
* **Best Result:** The tuning job achieved **100% accuracy** on the validation set using `entropy` and `n_estimators=7`.

### 5. Deployment (`serve.py`)
The best model is deployed as a **Serverless Inference Endpoint**.
* **Config:** 1024 MB Memory, Max Concurrency 4.
* **Custom Inference Script:** `serve.py` handles JSON input parsing, model loading, and probability output formatting.

## 💻 Usage

### Prerequisites
Ensure you have an AWS account with SageMaker permissions and the necessary Python libraries installed.

```bash
pip install -r requirements.txt