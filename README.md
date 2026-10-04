# 📉 Customer Churn Prediction

An end-to-end **machine learning pipeline for predicting telecom customer churn**, with SQL-based feature engineering, XGBoost modeling, cross-validation, and an interactive Streamlit application for both single-customer and batch predictions.

The system identifies customers who are likely to leave and provides **SHAP-based explanations** to show which factors are driving each prediction, enabling targeted customer-retention strategies.

## 🎯 Objective

Customer churn is a major challenge for telecom companies because retaining an existing customer is often more valuable than acquiring a new one.

This project uses historical customer data to:

- Predict whether a customer is likely to churn.
- Identify the strongest factors contributing to churn.
- Help businesses target high-risk customers with retention offers.
- Provide interpretable predictions through SHAP explanations.

## 📊 Model Performance

The model uses **XGBoost** with class-imbalance handling and was evaluated using 5-fold cross-validation.

- **ROC-AUC:** **0.835**
- **Recall on churners:** **74%** at the 0.5 classification threshold.

A recall of 74% means the model correctly identifies roughly **3 out of every 4 customers who actually churn**, making it useful for proactive retention campaigns.

### Key Churn Drivers

The strongest factors associated with customer churn were:

- **Month-to-month contracts**
- **Low customer tenure**
- **Electronic-check payment method**
- **Fiber-optic internet service**

## 🏗️ Machine Learning Pipeline

The project follows an end-to-end data-to-prediction workflow:

```text
Raw Customer Data
       ↓
SQL Feature Engineering
       ↓
Data Cleaning & Feature Creation
       ↓
One-Hot Encoding
       ↓
XGBoost Training
       ↓
5-Fold Cross-Validation
       ↓
Model Evaluation
       ↓
Streamlit Prediction App
       ↓
SHAP Explanations
```

### 1. SQL Feature Engineering

Feature engineering is performed using **DuckDB** through:

```text
sql/features.sql
```

Data cleaning and feature creation are performed inside the database before the data is passed to the machine learning pipeline.

### 2. Feature Encoding

Categorical variables are transformed using **one-hot encoding** in:

```text
src/features.py
```

This converts categorical customer information into numerical features suitable for model training.

### 3. Model Training

The project uses **XGBoost** for binary churn classification.

The training pipeline:

- Handles class imbalance using `scale_pos_weight`.
- Trains the XGBoost classifier.
- Evaluates performance using ROC-AUC.
- Saves the trained model for deployment.

Training code:

```text
src/train.py
```

### 4. Model Validation & Tuning

The model is evaluated using **5-fold cross-validation**.

Grid search is used to explore model hyperparameters and identify a stronger configuration.

Validation and tuning code:

```text
src/tune.py
```

### 5. Explainable Predictions

**SHAP (SHapley Additive exPlanations)** is used to explain individual predictions.

For each customer, the application can show which features contributed most strongly toward the predicted churn probability.

This makes the model more interpretable and helps connect predictions to actionable business decisions.

## 🖥️ Streamlit Application

The project includes an interactive **Streamlit** application supporting:

- Single-customer churn prediction.
- Batch predictions.
- Churn probability.
- SHAP-based prediction explanations.
- Easy-to-use interactive interface.

Launch the application with:

```bash
streamlit run app.py
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Machine learning pipeline |
| **Pandas** | Data manipulation |
| **SQL** | Feature engineering |
| **DuckDB** | In-database data processing |
| **XGBoost** | Churn prediction model |
| **SHAP** | Model explainability |
| **Scikit-learn** | Encoding, validation & evaluation |
| **Streamlit** | Interactive web application |

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── requirements.txt
│
├── sql/
│   └── features.sql
│
├── src/
│   ├── features.py
│   ├── train.py
│   └── tune.py
│
└── README.md
```

## 🚀 Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Customer-Churn-Prediction
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

**Mac/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python -m src.train
```

This trains the XGBoost model and saves the resulting model artifacts used by the application.

### 5. Launch the Streamlit application

```bash
streamlit run app.py
```

## 📂 Dataset

The project uses the **IBM Telco Customer Churn dataset**.

- **Customers:** 7,043
- **Features:** 21
- **Target:** Customer churn

The dataset contains information about customer demographics, account information, services, payment methods, tenure, and churn status.

## 💡 Business Impact

The goal of this project is not simply to predict churn, but to make the predictions **actionable**.

By identifying high-risk customers and explaining the factors behind each prediction, a telecom company could:

- Prioritize customers for retention campaigns.
- Offer targeted discounts or plan upgrades.
- Identify contract types associated with higher churn.
- Understand customer behaviors linked to churn.
- Allocate retention resources more efficiently.

## 📌 Key Takeaways

- Built an **end-to-end churn prediction pipeline** from SQL feature engineering to deployment.
- Used **XGBoost** with class-imbalance handling.
- Achieved **0.835 ROC-AUC** using 5-fold cross-validation.
- Achieved **74% recall on churners** at the 0.5 threshold.
- Used **SHAP** to make individual predictions interpretable.
- Deployed predictions through an interactive **Streamlit application**.
