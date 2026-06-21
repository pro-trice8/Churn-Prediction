import pandas as pd

def build_features(df):
    df = df.copy()
    target = df.pop("Churn")

    # One-hot encode all text columns into numeric 0/1 columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df["Churn"] = target.values
    return df