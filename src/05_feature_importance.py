import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES_PATH = "data/features_daily.csv"
MODEL_REG_PATH = "outputs/models/tmax_regression.joblib"
MODEL_CLF_PATH = "outputs/models/rain_classifier.joblib"
OUT_DIR = "outputs/plots"

def get_feature_names(preprocessor):
    # ColumnTransformer: ("num", ... num_features), ("cat", ... OneHotEncoder on ["city"])
    num_features = preprocessor.transformers_[0][2]
    ohe = preprocessor.transformers_[1][1].named_steps["ohe"]
    cat_features = preprocessor.transformers_[1][2]
    cat_names = list(ohe.get_feature_names_out(cat_features))
    return list(num_features) + cat_names

def save_importance_plot(names, importances, title, out_path, top_n=20):
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [names[i] for i in idx][::-1]
    top_vals = [importances[i] for i in idx][::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(top_names, top_vals)
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    reg_pipe = joblib.load(MODEL_REG_PATH)
    clf_pipe = joblib.load(MODEL_CLF_PATH)

    pre = reg_pipe.named_steps["pre"]
    feature_names = get_feature_names(pre)

    reg_model = reg_pipe.named_steps["model"]
    clf_model = clf_pipe.named_steps["model"]

    save_importance_plot(
        feature_names,
        reg_model.feature_importances_,
        "Feature Importance (Regression): Predict Tmax Tomorrow",
        f"{OUT_DIR}/feature_importance_regression.png",
        top_n=20
    )

    save_importance_plot(
        feature_names,
        clf_model.feature_importances_,
        "Feature Importance (Classification): Predict Rain Tomorrow",
        f"{OUT_DIR}/feature_importance_classification.png",
        top_n=20
    )

    reg_top = sorted(zip(feature_names, reg_model.feature_importances_), key=lambda x: x[1], reverse=True)[:15]
    clf_top = sorted(zip(feature_names, clf_model.feature_importances_), key=lambda x: x[1], reverse=True)[:15]

    print("\nTop 15 features (Regression - Tmax Tomorrow):")
    for n, v in reg_top:
        print(f"{n:30s} {v:.4f}")

    print("\nTop 15 features (Classification - Rain Tomorrow):")
    for n, v in clf_top:
        print(f"{n:30s} {v:.4f}")

    print(f"\nSaved plots -> {OUT_DIR}\\feature_importance_regression.png")
    print(f"Saved plots -> {OUT_DIR}\\feature_importance_classification.png")

if __name__ == "__main__":
    main()
