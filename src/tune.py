from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from xgboost import XGBClassifier

from src.data_prep import load_data
from src.features import build_features


def tune():
    df = build_features(load_data())
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    spw = (y == 0).sum() / (y == 1).sum()

    base = XGBClassifier(
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=42,
    )

    # 5-fold cross-validation on a reasonable baseline
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(base, X, y, cv=cv, scoring="roc_auc")
    print("5-fold CV ROC-AUC:", scores.round(4))
    print("Mean:", round(scores.mean(), 4), "+/-", round(scores.std(), 4))

    # Small grid search for the best hyperparameters
    grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    search = GridSearchCV(base, grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1)
    search.fit(X, y)

    print("\nBest ROC-AUC:", round(search.best_score_, 4))
    print("Best params:", search.best_params_)


if __name__ == "__main__":
    tune()