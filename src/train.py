import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

from src.data_prep import load_data
from src.features import build_features

def train():
    df = build_features((load_data()))

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Weight the minority (churn) class so the model doesn't ignore it
    spw = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))

    # Save BOTH the model and the exact column order
    joblib.dump({"model": model, "columns": X.columns.tolist()},
                "models/churn_model.pkl")
    print("Saved model to models/churn_model.pkl")

if __name__ == "__main__":
    train()