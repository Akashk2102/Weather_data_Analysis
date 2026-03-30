import os
import json
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import joblib

DATA_PATH = "data/features_daily.csv"
METRICS_DIR = "outputs/metrics"
MODELS_DIR = "outputs/models"

def time_split(df: pd.DataFrame, test_size=0.2):
    df = df.sort_values("time")
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run src/02_prepare_features.py first.")

    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])

    target_reg = "tmax_tomorrow"
    target_clf = "rain_tomorrow"

    data = df.drop(columns=[c for c in ["prcp_tomorrow"] if c in df.columns]).copy()

    train_df, test_df = time_split(data, test_size=0.2)

    X_train = train_df.drop(columns=[target_reg, target_clf])
    X_test  = test_df.drop(columns=[target_reg, target_clf])

    y_reg_train = train_df[target_reg]
    y_reg_test  = test_df[target_reg]

    y_clf_train = train_df[target_clf]
    y_clf_test  = test_df[target_clf]

    if "time" in X_train.columns:
        X_train = X_train.drop(columns=["time"])
        X_test = X_test.drop(columns=["time"])

    cat_features = ["city"]
    num_features = [c for c in X_train.columns if c not in cat_features]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_features),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_features),
        ]
    )

    reg = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    reg_pipe = Pipeline([("pre", pre), ("model", reg)])
    reg_pipe.fit(X_train, y_reg_train)
    reg_pred = reg_pipe.predict(X_test)

    reg_mae = mean_absolute_error(y_reg_test, reg_pred)
    reg_rmse = mean_squared_error(y_reg_test, reg_pred) ** 0.5
    reg_r2 = r2_score(y_reg_test, reg_pred)

    joblib.dump(reg_pipe, f"{MODELS_DIR}/tmax_regression.joblib")

    clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced")
    clf_pipe = Pipeline([("pre", pre), ("model", clf)])
    clf_pipe.fit(X_train, y_clf_train)

    clf_pred = clf_pipe.predict(X_test)
    clf_proba = clf_pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_clf_test, clf_pred)
    f1 = f1_score(y_clf_test, clf_pred)

    try:
        auc = roc_auc_score(y_clf_test, clf_proba)
    except ValueError:
        auc = None

    cm = confusion_matrix(y_clf_test, clf_pred).tolist()

    joblib.dump(clf_pipe, f"{MODELS_DIR}/rain_classifier.joblib")

    metrics = {
        "regression": {"MAE": reg_mae, "RMSE": reg_rmse, "R2": reg_r2},
        "classification": {"Accuracy": acc, "F1": f1, "ROC_AUC": auc, "ConfusionMatrix": cm},
        "split": {"type": "time-based", "train_rows": int(len(train_df)), "test_rows": int(len(test_df))}
    }

    with open(f"{METRICS_DIR}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics -> {METRICS_DIR}/metrics.json")
    print(metrics)

if __name__ == "__main__":
    main()
