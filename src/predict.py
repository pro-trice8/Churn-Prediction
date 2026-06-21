import joblib
from src.features import build_features

def load_model(path="models/churn_model.pkl"):
    bundle = joblib.load(path)
    return bundle["model"], bundle["columns"]

def predict(raw_df, model=None, columns=None):
    if model is None:
        model, columns = load_model()
    df = build_features(raw_df)
    df = df.drop(columns=["Churn"], errors="ignore")
    df = df.reindex(columns=columns, fill_value=0)
    return model.predict_proba(df)[:, 1]