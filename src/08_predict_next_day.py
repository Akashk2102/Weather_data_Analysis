import os
from datetime import timedelta
import pandas as pd
import joblib

PRED_INPUT = "data/predict_today_input.csv"  # contains latest available row per city
REG_MODEL_PATH = "outputs/models/tmax_regression.joblib"
CLF_MODEL_PATH = "outputs/models/rain_classifier.joblib"
OUT_DIR = "outputs/predictions"
OUT_PATH = f"{OUT_DIR}/predictions_next_day.csv"

def main():
    if not os.path.exists(PRED_INPUT):
        raise FileNotFoundError(f"Missing {PRED_INPUT}. Run src/02_prepare_features.py first.")

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(PRED_INPUT)
    df["time"] = pd.to_datetime(df["time"])

    observed_asof = df["time"].copy()
    predicted_for = (df["time"] + pd.Timedelta(days=1))

    # Model features: drop time (same as training)
    X = df.drop(columns=["time"]) if "time" in df.columns else df.copy()

    reg_pipe = joblib.load(REG_MODEL_PATH)
    clf_pipe = joblib.load(CLF_MODEL_PATH)

    out = df[["city"]].copy()
    out["observed_asof_date"] = observed_asof.dt.date
    out["predicted_for_date"] = predicted_for.dt.date
    out["pred_tmax"] = reg_pipe.predict(X)
    out["pred_rain"] = clf_pipe.predict(X)
    out["proba_rain"] = clf_pipe.predict_proba(X)[:, 1]

    out = out.sort_values("city")
    out.to_csv(OUT_PATH, index=False)

    print(f"Saved -> {OUT_PATH}")
    print(out)

if __name__ == "__main__":
    main()
