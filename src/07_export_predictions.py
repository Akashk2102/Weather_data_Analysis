import os
import pandas as pd
import joblib

DATA_PATH = "data/features_daily.csv"
REG_MODEL_PATH = "outputs/models/tmax_regression.joblib"
CLF_MODEL_PATH = "outputs/models/rain_classifier.joblib"
OUT_DIR = "outputs/predictions"
OUT_PATH = f"{OUT_DIR}/test_predictions.csv"

def time_split(df, test_size=0.2):
    df = df.sort_values("time")
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])

    # Same leakage drop as training
    data = df.drop(columns=[c for c in ["prcp_tomorrow"] if c in df.columns]).copy()
    _, test_df = time_split(data, test_size=0.2)

    target_reg = "tmax_tomorrow"
    target_clf = "rain_tomorrow"

    X_test = test_df.drop(columns=[target_reg, target_clf]).copy()
    if "time" in X_test.columns:
        X_feat = X_test.drop(columns=["time"])
    else:
        X_feat = X_test

    reg_pipe = joblib.load(REG_MODEL_PATH)
    clf_pipe = joblib.load(CLF_MODEL_PATH)

    pred_tmax = reg_pipe.predict(X_feat)
    pred_rain = clf_pipe.predict(X_feat)
    proba_rain = clf_pipe.predict_proba(X_feat)[:, 1]

    out = pd.DataFrame({
        "time": test_df["time"].values,
        "city": test_df["city"].values,
        "actual_tmax_tomorrow": test_df[target_reg].values,
        "pred_tmax_tomorrow": pred_tmax,
        "actual_rain_tomorrow": test_df[target_clf].values,
        "pred_rain_tomorrow": pred_rain,
        "proba_rain_tomorrow": proba_rain,
    }).sort_values(["city", "time"])

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
    print(out.head(10))

if __name__ == "__main__":
    main()
