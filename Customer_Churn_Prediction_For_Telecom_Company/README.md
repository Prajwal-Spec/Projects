**Telecom Customer Churn Prediction Report**

**1. Introduction**
This project aimed to predict customer churn using telecom service data. A Random Forest classifier was trained and tuned to identify key factors influencing churn and to provide actionable business recommendations.

**2. Data Preprocessing**
- Handled non-numeric values in TotalCharges by conversion and median imputation.

- Encoded categorical variables using label encoding and one-hot encoding.

- Scaled numerical features for consistency.

- Created new features indicating multiple lines and streaming service usage.

- Dropped redundant columns to reduce noise.

**3. Feature Engineering**
- Added binary flags has_multiple_lines and has_streaming.

- Initially explored tenure grouping, but excluded due to scaling conflicts.

**4. Data Splitting**
- Split data into 80% training and 20% testing with stratification on the churn target.

- Customer IDs were removed from feature sets to prevent overfitting.

**5. Model Selection and Training**
- Random Forest classifier selected for classification task.

- Hyperparameter tuning applied using GridSearchCV to optimize depth, splits, and class weights.

- Obtained training accuracy of ~85% and testing accuracy of ~80%.

**6. Model Evaluation**
- Metrics on test data:

  - Accuracy: 80.5%

  - Precision: 66.9%

  - Recall: 52.9%

  - F1-Score: 59.1%

  - ROC-AUC: 84.1%

- Confusion matrix indicated balanced false positive and false negative rates.

- Model is effective at identifying churners with reasonable trade-offs.

**7. Insights and Recommendations**
- Important features: Tenure, Contract Type, Total Charges, Tech Support, Internet Service.

- Customers on month-to-month contracts and shorter tenure are high risk.

**Suggestions:**

- Targeted retention campaigns for short-term customers.

- Incentivize longer contracts.

- Improve tech support and online security services.

- Review pricing and payment methods for at-risk customers.

**8. Code and Visualization**
- Code sections for preprocessing, model training, tuning, and evaluation documented.

- Feature importance visualized with a custom gradient bar chart.

---

For questions and contributions, contact kumarprajwal333@gmail.com

