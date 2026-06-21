# Customer Churn Prediction

End-to-end machine learning pipeline that predicts which telecom customers
are likely to churn, with an interactive Streamlit app for single and
batch predictions. Features are engineered in SQL; the model is XGBoost.

## Results

- ROC-AUC: **0.835** (5-fold cross-validated)
- Recall on churners: **74%** at the 0.5 threshold — the model catches
  roughly three of every four customers who would actually leave,
  enabling targeted retention offers before they go.
- Strongest churn drivers: month-to-month contracts, low tenure,
  electronic-check payments, and fiber-optic internet.

## Tech stack

Python, Pandas, SQL (DuckDB), XGBoost, SHAP, Streamlit.

## Pipeline

1. **SQL feature engineering** (`sql/features.sql`) — cleaning and
   feature creation run in the database; the model trains on the output.
2. **Encoding** (`src/features.py`) — one-hot encoding for the model.
3. **Training** (`src/train.py`) — XGBoost with class-imbalance handling
   (`scale_pos_weight`), evaluated with ROC-AUC.
4. **Validation** (`src/tune.py`) — 5-fold cross-validation and grid search.
5. **App** (`app.py`) — Streamlit UI with per-prediction SHAP explanations.

## Run it

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt

python -m src.train              # train and save the model
streamlit run app.py             # launch the app
```

## Dataset

IBM Telco Customer Churn (7,043 customers, 21 features).