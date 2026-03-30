import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import confusion_matrix, roc_curve, auc

DATA_PATH = "data/features_daily.csv"
REG_MODEL_PATH = "outputs/models/tmax_regression.joblib"
CLF_MODEL_PATH = "outputs/models/rain_classifier.joblib"
OUT_DIR = "outputs/plots"

def time_split(df, test_size=0.2):
    df = df.sort_values("time")
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sns.set_style("whitegrid")

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])

    # Keep same setup as training
    data = df.drop(columns=[c for c in ["prcp_tomorrow"] if c in df.columns]).copy()
    train_df, test_df = time_split(data, test_size=0.2)

    target_reg = "tmax_tomorrow"
    target_clf = "rain_tomorrow"

    X_test = test_df.drop(columns=[target_reg, target_clf]).copy()
    y_reg = test_df[target_reg].values
    y_clf = test_df[target_clf].values

    if "time" in X_test.columns:
        X_test = X_test.drop(columns=["time"])

    reg_pipe = joblib.load(REG_MODEL_PATH)
    clf_pipe = joblib.load(CLF_MODEL_PATH)

    # ---------- Regression: actual vs predicted ----------
    reg_pred = reg_pipe.predict(X_test)

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(y_reg, reg_pred, s=10, alpha=0.4)
    mn = float(min(y_reg.min(), reg_pred.min()))
    mx = float(max(y_reg.max(), reg_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.title("Regression: Actual vs Predicted (Tmax Tomorrow)")
    plt.xlabel("Actual Tmax Tomorrow (°C)")
    plt.ylabel("Predicted Tmax Tomorrow (°C)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/reg_actual_vs_pred.png", dpi=200)
    plt.close()

    # ---------- Classification: confusion matrix ----------
    clf_pred = clf_pipe.predict(X_test)
    cm = confusion_matrix(y_clf, clf_pred)

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Classification: Confusion Matrix (Rain Tomorrow)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/clf_confusion_matrix.png", dpi=200)
    plt.close()

    # ---------- Classification: ROC curve ----------
    proba = clf_pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_clf, proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6.5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Classification: ROC Curve (Rain Tomorrow)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/clf_roc_curve.png", dpi=200)
    plt.close()

    print("Saved:")
    print(f" - {OUT_DIR}/reg_actual_vs_pred.png")
    print(f" - {OUT_DIR}/clf_confusion_matrix.png")
    print(f" - {OUT_DIR}/clf_roc_curve.png")

if __name__ == "__main__":
    main()
