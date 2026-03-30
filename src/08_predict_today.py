import os
import pandas as pd
import joblib

PRED_INPUT = "data/predict_today_input.csv"
REG_MODEL_PATH = "outputs/models/tmax_regression.joblib"
CLF_MODEL_PATH = "outputs/models/rain_classifier.joblib"
OUT_PATH = "outputs/predictions/predictions_for_today.csv"

def main():
    if not os.path.exists(PRED_INPUT):
        raise FileNotFoundError(f"Missing {PRED_INPUT}. Run src/02_prepare_features.py first.")

    os.makedirs("outputs/predictions", exist_ok=True)

    df = pd.read_csv(PRED_INPUT)
    df["time"] = pd.to_datetime(df["time"])

    # Drop time column for model (same as training)
    X = df.drop(columns=["time"]) if "time" in df.columns else df.copy()

    reg_pipe = joblib.load(REG_MODEL_PATH)
    clf_pipe = joblib.load(CLF_MODEL_PATH)

    df["pred_tmax_today"] = reg_pipe.predict(X)
    df["pred_rain_today"] = clf_pipe.predict(X)
    df["proba_rain_today"] = clf_pipe.predict_proba(X)[:, 1]

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
    df["predicted_for_date"] = (pd.to_datetime(df["time"]) + pd.Timedelta(days=1)).dt.date
    print(df[["city","time","predicted_for_date","pred_tmax_today","pred_rain_today","proba_rain_today"]])

if __name__ == "__main__":
    main()
