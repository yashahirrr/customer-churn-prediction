
# Customer Churn Prediction

An end-to-end machine learning application that predicts customer churn probability and provides model explanations using SHAP.

## Live Demo

Coming soon — deployed using Streamlit Community Cloud.

## Project Overview

Customer churn is an important problem for subscription-based businesses because retaining existing customers can be more cost-effective than acquiring new customers.

This project develops a machine learning system that predicts the probability of customer churn using customer demographics, subscribed services, account information, and billing details.

The project covers the complete machine learning workflow:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Class imbalance handling
- Model training and comparison
- Threshold optimization
- Model evaluation
- SHAP-based model explainability
- Interactive Streamlit deployment

The application allows users to enter customer information and receive a churn probability, risk level, and explanation of the prediction.

## Dataset

The project uses the Telco Customer Churn dataset.

Dataset source:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

- 7,043 customer records
- 21 features
- Binary target variable: `Churn`

### Feature Categories

**Customer Demographics**

- Gender
- Senior Citizen
- Partner
- Dependents

**Account Information**

- Tenure
- Contract
- Paperless Billing
- Payment Method

**Services**

- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

**Billing**

- Monthly Charges
- Total Charges

## Machine Learning Pipeline

The preprocessing pipeline is implemented using Scikit-learn's `ColumnTransformer`.

### Numerical Features

- Tenure
- Monthly Charges
- Total Charges

Processing:

- Missing value imputation using the median
- Standard scaling using `StandardScaler`

### Categorical Features

Categorical features are processed using:

- Most-frequent-value imputation
- One-hot encoding
- `handle_unknown='ignore'`

The preprocessing steps are integrated into the model pipelines so that the same transformations are applied during both training and prediction.

## Models

Three classification models are trained and compared.

### Logistic Regression

Used as an interpretable baseline model with class-balanced training.

### Random Forest

An ensemble model capable of capturing non-linear relationships between customer characteristics and churn.

Configuration includes class-balanced training and 200 trees.

### XGBoost

A gradient boosting model configured with:

- 200 estimators
- Learning rate of 0.05
- Maximum depth of 4
- Subsampling
- Feature subsampling
- `scale_pos_weight` for class imbalance
- Log-loss evaluation metric

## Model Performance

The deployed application evaluates the models using the saved test dataset.

| Model               | ROC-AUC | F1 Score | Precision | Recall |
| ------------------- | ------: | -------: | --------: | -----: |
| Logistic Regression |    0.84 |     0.62 |      0.51 |   0.79 |
| Random Forest       |    0.89 |     0.69 |      0.60 |   0.80 |
| XGBoost             |    0.88 |     0.67 |      0.56 |   0.84 |

The application provides additional evaluation through:

- ROC curves
- Confusion matrices
- Model comparison
- Feature importance

## Threshold Optimization

The project evaluates classification thresholds beyond the default 0.5 threshold.

Changing the threshold allows the trade-off between precision and recall to be adjusted depending on the desired business strategy.

For churn prediction, this can be useful when the cost of missing a potential churn customer differs from the cost of contacting a customer who ultimately does not churn.

## Model Explainability

The application uses SHAP (SHapley Additive exPlanations) to provide model explanations.

SHAP is used for:

- Individual prediction explanations
- Global feature importance
- Understanding which features contribute to predictions

The Streamlit application displays SHAP explanations for individual customer predictions as well as global SHAP visualizations.

### Model-Identified Churn Drivers

The model identifies several features as important predictors of churn, including:

- Customer tenure
- Contract type
- Monthly charges
- Internet service
- Value-added services such as Online Security and Tech Support

These should be interpreted as model associations rather than causal relationships.

## Business Insights

The model can help identify customer segments associated with higher predicted churn risk.

Examples of potential retention strategies include:

- Targeting customers with high predicted churn probability
- Focusing attention on newer customers
- Evaluating customers on month-to-month contracts
- Reviewing customers with relatively high monthly charges
- Considering service bundles and retention offers

These recommendations are intended as examples of how model predictions could support business decision-making.

## Streamlit Application

The application provides two main sections.

### Prediction

Users can:

- Select a machine learning model
- Enter customer information
- Generate churn probability
- View the predicted risk level
- View the model used
- Inspect the SHAP explanation for the prediction

### Model Insights

Users can explore:

- Model performance
- ROC curves
- Confusion matrices
- Model comparison
- Feature importance
- SHAP summary plots
- SHAP feature importance
- Business insights

## Project Structure

```text
Customer-Churn-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── Telco_Customer_Churn.csv
│   ├── X_test.csv
│   └── y_test.csv
│
├── models/
│   ├── logistic_model.pkl
│   ├── rf_model.pkl
│   └── xgb_model.pkl
│
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```
