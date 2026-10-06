
# Telco Customer Churn Prediction

## Project Overview

Customer churn represents customers discontinuing their relationship with a service provider. Identifying customers who are likely to churn can help a business take proactive retention measures.

The objective of this project is to develop a machine learning model that can identify customers at risk of churn while also investigating the customer, service, contract, and billing characteristics associated with churn.

Because failing to identify a customer who eventually churns may be more costly than incorrectly flagging a customer who remains, **recall for the churn class is treated as the primary evaluation metric**. Precision, F1-score, and the confusion matrix are also considered to ensure that improvements in recall do not result in an unreasonable number of false positives.

---

## Dataset

The dataset contains **7,032 customer records and 21 variables** describing customer demographics, subscribed services, contract information, billing characteristics, and churn status.

Initial data inspection showed:

- No missing values
- No duplicate records
- Each `customerID` is unique
- A combination of numerical and categorical features
- Approximately **27% of customers churned**, compared with approximately **73% who remained**

`customerID` was treated only as an identifier and was excluded from predictive modeling.

---

## Project Workflow

The project follows an end-to-end machine learning workflow:

1. Data loading and initial inspection
2. Exploratory Data Analysis (EDA)
3. Feature selection and relationship analysis
4. Data preprocessing
5. Model development and baseline evaluation
6. Model comparison and hyperparameter tuning
7. Final test-set evaluation
8. Model interpretation
9. Business recommendations

---

## Exploratory Data Analysis

Exploratory analysis was performed to understand the distribution of churn and investigate relationships between customer characteristics and churn behavior.

### Numerical Features

Customers who churned generally had:

- **Shorter tenure**
- **Higher monthly charges**
- **Lower total charges**, which was consistent with their shorter average tenure

The relationship between `MonthlyCharges` and churn was also investigated within different internet-service groups. This showed that overall relationships can change after conditioning on another variable, highlighting the importance of looking beyond simple aggregate comparisons.

### Categorical Features

Several categorical variables showed notable differences in churn rates.

Higher churn rates were observed among:

- Month-to-month contract customers
- Fiber-optic internet customers
- Customers using electronic-check payments
- Customers using paperless billing
- Customers without online security
- Customers without technical support

Longer-term contracts, online security, and technical support were associated with lower churn rates.

---

## Feature Selection and Relationship Analysis

Feature selection combined statistical testing, exploratory analysis, redundancy analysis, and model-validation considerations rather than relying on a single method.

### Chi-Square Analysis

Chi-Square tests were used to investigate associations between categorical predictors and churn.

Variables such as:

- Contract
- Internet service
- Online security
- Technical support
- Payment method
- Online backup
- Device protection
- Paperless billing

showed significant associations with churn.

Statistical significance was not used as the only criterion for feature selection. Practical differences, structural relationships, and redundancy between variables were also considered.

### Conditional Relationship Analysis

Several relationships were examined within other customer groups to determine whether their association with churn remained after accounting for related variables.

For example, internet add-on services were examined within internet-service categories, while partner and dependent status were considered alongside contract type.

This helped distinguish useful relationships from patterns that could largely be explained by other variables.

### Multicollinearity Analysis

Correlation analysis and **Variance Inflation Factor (VIF)** were used to investigate multicollinearity among numerical predictors.

`TotalCharges` showed substantial overlap with `tenure` and `MonthlyCharges`. Removing `TotalCharges` substantially reduced the VIF of the remaining numerical predictors.

For this reason, `TotalCharges` was excluded from the final predictor set.

### Other Feature Decisions

- `customerID` was excluded because it is an identifier.
- `gender` was excluded because it showed very little relationship with churn.
- `PhoneService` was excluded because its information was already represented through `MultipleLines`.
- `SeniorCitizen` was retained as a binary predictor.
- Several weaker features were retained initially so their usefulness could be assessed during model development.

The final modeling dataset contained **16 predictor variables**.

---

## Data Preprocessing

Before modeling, the structural `"No internet service"` category in internet add-on variables was simplified to `"No"` because `InternetService` already identifies customers without internet service.

The target variable was encoded as:

- `1` = customer churned
- `0` = customer remained

The data was divided into:

- **70% training data**
- **15% validation data**
- **15% test data**

Stratified splitting was used to preserve approximately the same churn proportion across the three subsets.

Two preprocessing pipelines were created.

For Logistic Regression:

- Categorical features were one-hot encoded
- The first categorical level was dropped as a reference category
- Continuous numerical variables were standardized
- Binary variables were passed through unchanged

For tree-based models:

- Categorical variables were one-hot encoded
- Numerical variables were retained without scaling

Keeping preprocessing within Scikit-learn pipelines ensured that transformations were learned from the appropriate training data.

---

## Model Development

Four classification approaches were evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Balanced Logistic Regression

The models were initially evaluated using the default classification threshold of `0.50`.

Because identifying actual churners was the primary objective, **recall** was emphasized during model comparison. Precision, F1-score, and confusion matrices were also examined.

---

## Classification Threshold Analysis

Different probability thresholds were evaluated because the default threshold of `0.50` does not necessarily provide the most suitable trade-off for churn detection.

Lower thresholds generally increased recall because more customers were classified as potential churners. However, this also reduced precision by increasing the number of false-positive predictions.

Threshold analysis therefore helped evaluate the trade-off between:

- Detecting more actual churners
- Avoiding excessive false-positive churn predictions

---

## Model Selection

Balanced Logistic Regression provided the strongest recall among the candidate approaches while maintaining a reasonable precision-recall trade-off.

It was therefore selected as the **candidate model for hyperparameter tuning**.

This choice reflects the project's objective: missing an actual churner was considered more costly than incorrectly identifying a non-churner as being at risk.

---

## Hyperparameter Tuning

The selected Balanced Logistic Regression model was tuned using **GridSearchCV** with **5-fold Stratified Cross-Validation**.

The Logistic Regression regularization parameter `C` was evaluated using:

```python
[0.01, 0.1, 1, 10, 100]
```

Recall was used as the optimization metric.

Hyperparameter tuning was performed using only the training data. The validation and test sets were not used during cross-validation.

The tuning process confirmed that the existing regularization configuration was already suitable for the problem.

---

## Final Model Evaluation

After model selection, hyperparameter tuning, and threshold selection were completed, the final Balanced Logistic Regression model was evaluated on the previously untouched test set.

The final classification threshold was fixed at **0.50** before test-set evaluation.

### Test Performance

| Metric    |          Result |
| --------- | --------------: |
| Recall    | **75.4%** |
| Precision | **48.8%** |
| F1-score  | **59.3%** |

The final confusion matrix was:

```text
[[554, 221],
 [ 69, 211]]
```

Of the **280 actual churners** in the test set:

- **211 were correctly identified**
- **69 were missed**

The model therefore identified approximately **3 out of every 4 customers who actually churned**.

The model also incorrectly classified **221 non-churning customers** as potential churners.

The lower precision represents the trade-off made by prioritizing recall. The model was designed to identify a larger proportion of customers who may be at risk, even if this results in additional false-positive predictions.

---

## Model Interpretation

Because the final model is Logistic Regression, its coefficients were examined to understand which customer characteristics contributed most strongly to predicted churn.

### Characteristics Associated with Higher Predicted Churn

Important positive relationships included:

- Fiber-optic internet service
- Electronic-check payment
- Paperless billing
- Some streaming-service characteristics

### Characteristics Associated with Lower Predicted Churn

Important negative relationships included:

- Two-year contracts
- One-year contracts
- Longer tenure
- Online security
- Technical support
- Having dependents

Contract type, tenure, internet service, support services, payment method, and billing characteristics therefore emerged as important contributors to the final model's churn predictions.

These relationships represent **associations within the fitted model while holding other included features constant**. They should not be interpreted as evidence that any individual characteristic directly causes churn.

---

## Business Recommendations

Based on the exploratory analysis and final predictive model, several areas may be useful for customer-retention strategies.

### Prioritize Month-to-Month Customers

Longer-term contracts, particularly two-year contracts, were associated with substantially lower predicted churn. Customers remaining on month-to-month contracts may therefore represent an important group for targeted retention efforts.

### Focus on Newer Customers

Longer tenure was associated with lower predicted churn. Customers in the earlier stages of their relationship with the company may benefit from additional engagement and retention support.

### Investigate Fiber-Optic Customer Experience

Fiber-optic internet service was associated with higher predicted churn even after other model variables were considered.

Further investigation into factors such as pricing, service quality, and customer satisfaction may help explain this relationship.

### Examine Support Services

Online security and technical support were associated with lower predicted churn.

These services may therefore be useful areas for further investigation when developing customer-retention strategies.

### Use Billing Characteristics as Risk Indicators

Electronic-check payment and paperless billing were associated with higher predicted churn and may provide additional information when identifying customers requiring attention.

These findings identify customer segments and risk indicators rather than establishing causal relationships. Further business investigation or controlled experiments would be required before concluding that changing any individual characteristic would directly reduce churn.

---

## Repository Structure

```text
Customer_Churn/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebook/
│   └── Churn_Project_Notebook.ipynb
│
├── Telco_Churn_Project_Brief.pdf
├── README.md
├── requirements.txt
└── .gitignore
```

> **Note:** Update `Churn_Project_Notebook.ipynb` above if your notebook has a different filename.

---

## Technologies Used

- **Python** — programming language
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical operations
- **Matplotlib** — data visualization
- **Seaborn** — statistical visualization
- **SciPy** — Chi-Square statistical testing
- **Statsmodels** — Variance Inflation Factor analysis
- **Scikit-learn** — preprocessing, machine learning, cross-validation, and evaluation
- **Jupyter Notebook** — interactive analysis and project documentation

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project directory

```bash
cd Customer_Churn
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

Open the notebook located inside the `notebook` directory using Jupyter Notebook, JupyterLab, or VS Code.

### 5. Run the notebook

Run the notebook cells from top to bottom.

The dataset is loaded using the relative path:

```python
../data/Telco-Customer-Churn.csv
```

Therefore, the repository folder structure should be preserved when running the project.

---

## Conclusion

This project developed an end-to-end machine learning workflow for identifying customers at risk of churn using demographic, service, contract, and billing information.

Exploratory analysis and statistical relationship analysis were first used to understand the factors associated with customer churn. Multiple classification algorithms were then evaluated, with recall prioritized because failing to identify an actual churner was considered more costly than incorrectly flagging a customer who remained.

Balanced Logistic Regression was selected and tuned using stratified cross-validation.

On previously unseen test data, the final model achieved:

- **75.4% recall**
- **48.8% precision**
- **59.3% F1-score**

The model correctly identified **211 of 280 actual churners**, demonstrating its ability to identify the majority of customers who eventually left.

The project also showed that contract type, tenure, internet service, support services, payment methods, and billing characteristics contribute meaningfully to churn predictions.

Overall, the final model can serve as a **customer churn screening tool**, helping a business prioritize customers for further investigation and retention efforts while recognizing the trade-off between identifying more churners and generating additional false-positive predictions.
