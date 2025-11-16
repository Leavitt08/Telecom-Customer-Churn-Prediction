# save_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import json
import os

CSV_PATH = "telecom_customer_churn.csv"
MODEL_OUT = "model.joblib"
FEATURES_OUT = "features.json"

def load_and_prepare():
    df = pd.read_csv(CSV_PATH)
    # Notebook preprocessing steps (mirrors your notebook)
    # Keep only Stayed/Churned and create binary target
    df_model = df[df["Customer Status"].isin(["Stayed","Churned"])].copy()
    df_model["Churn"] = (df_model["Customer Status"] == "Churned").astype(int)

    # some derived columns used in notebook
    if "Tenure in Months" in df_model.columns:
        df_model["Tenure in Months"] = df_model["Tenure in Months"].replace(0, np.nan)
    if "Total Revenue" in df_model.columns and "Tenure in Months" in df_model.columns:
        df_model["Revenue_per_Month"] = df_model["Total Revenue"] / df_model["Tenure in Months"]
        df_model["Revenue_per_Month"] = df_model["Revenue_per_Month"].fillna(0)

    # drop unwanted columns if present (reflects notebook)
    drop_cols = [c for c in ["Customer Status", "Churn Category", "Churn Reason", "Customer ID"] if c in df_model.columns]
    if drop_cols:
        df_model.drop(columns=drop_cols, inplace=True)

    target_col = "Churn"
    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    # identify numerical and categorical as in notebook
    numerical_feature = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_feature = X.select_dtypes(include=["object"]).columns.tolist()

    return X, y, numerical_feature, categorical_feature

def build_and_save():
    X, y, numerical_feature, categorical_feature = load_and_prepare()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_feature),
            ("cat", categorical_transformer, categorical_feature)
        ]
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    # train/test split as in notebook
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train, y_train)

    # Save model
    dump(clf, MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")

    # Save feature list and dtypes for the form
    features = []
    for col in X.columns:
        dtype = str(X[col].dtype)
        features.append({"name": col, "dtype": dtype})
    with open(FEATURES_OUT, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)
    print(f"Saved feature metadata to {FEATURES_OUT}")

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found. Place your CSV at this path.")
    build_and_save()
