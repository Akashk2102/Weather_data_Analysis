import os
import numpy as np
import pandas as pd
import joblib

PRED_INPUT = "data/predict_today_input.csv"  # latest available row per city
REG_MODEL_PATH = "outputs/models/tmax_regression.joblib"
CLF_MODEL_PATH = "outputs/models/rain_classifier.joblib"
OUT_DIR = "outputs/predictions"
OUT_PATH = f"{OUT_DIR}/forecast_next_7_days.csv"

HORIZON_DAYS = 7

def safe_get(row, col, default=np.nan):
    return row[col] if col in row.index else default

def main():
    if not os.path.exists(PRED_INPUT):
        raise FileNotFoundError(f"Missing {PRED_INPUT}. Run src/02_prepare_features.py first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    base = pd.read_csv(PRED_INPUT)
    base["time"] = pd.to_datetime(base["time"])

    reg_pipe = joblib.load(REG_MODEL_PATH)
    clf_pipe = joblib.load(CLF_MODEL_PATH)

    results = []

    for _, r in base.iterrows():
        city = r["city"]
        observed_asof = pd.to_datetime(r["time"]).date()

        # We'll maintain a rolling "state" for features needed by the model.
        # NOTE: Our feature set includes many columns, but only a few are crucial.
        # We update what we can (tmax and calendar); others remain as last known values.

        # last known values
        last_tmax = safe_get(r, "tmax", np.nan)
        last_tavg = safe_get(r, "tavg", np.nan)
        last_tmin = safe_get(r, "tmin", np.nan)
        last_prcp = safe_get(r, "prcp", 0.0)
        last_wspd = safe_get(r, "wspd", np.nan)
        last_pres = safe_get(r, "pres", np.nan)

        # lag placeholders (if they exist in row)
        tmax_lag1 = safe_get(r, "tmax_lag1", np.nan)
        tmax_lag2 = safe_get(r, "tmax_lag2", np.nan)
        tmax_lag3 = safe_get(r, "tmax_lag3", np.nan)

        # rolling placeholders (if they exist)
        tmax_roll7 = safe_get(r, "tmax_roll7", np.nan)
        tmax_roll30 = safe_get(r, "tmax_roll30", np.nan)

        prcp_lag1 = safe_get(r, "prcp_lag1", np.nan)
        prcp_roll7 = safe_get(r, "prcp_roll7", np.nan)
        prcp_roll30 = safe_get(r, "prcp_roll30", np.nan)

        # Start from the latest observed date and forecast forward
        current_time = pd.to_datetime(r["time"])

        for step in range(1, HORIZON_DAYS + 1):
            predicted_for = (current_time + pd.Timedelta(days=step))
            year = int(predicted_for.year)
            month = int(predicted_for.month)
            dayofyear = int(predicted_for.dayofyear)

            # Build one-row feature frame with the same columns as training-time X.
            # We'll reuse base columns (except time/targets), fill city + calendar,
            # and update a subset of lag/roll fields.
            feat = r.copy()

            # Set calendar for the date we are predicting FOR (helps seasonality)
            feat["year"] = year
            feat["month"] = month
            feat["dayofyear"] = dayofyear

            # Update core recent values (best-effort)
            feat["tmax"] = last_tmax
            if "tavg" in feat.index: feat["tavg"] = last_tavg
            if "tmin" in feat.index: feat["tmin"] = last_tmin
            if "prcp" in feat.index: feat["prcp"] = last_prcp
            if "wspd" in feat.index: feat["wspd"] = last_wspd
            if "pres" in feat.index: feat["pres"] = last_pres

            # Update lags if present
            if "tmax_lag1" in feat.index: feat["tmax_lag1"] = tmax_lag1
            if "tmax_lag2" in feat.index: feat["tmax_lag2"] = tmax_lag2
            if "tmax_lag3" in feat.index: feat["tmax_lag3"] = tmax_lag3

            if "prcp_lag1" in feat.index: feat["prcp_lag1"] = prcp_lag1

            # Update rolling if present (approx; we don't have full window history)
            if "tmax_roll7" in feat.index: feat["tmax_roll7"] = tmax_roll7
            if "tmax_roll30" in feat.index: feat["tmax_roll30"] = tmax_roll30
            if "prcp_roll7" in feat.index: feat["prcp_roll7"] = prcp_roll7
            if "prcp_roll30" in feat.index: feat["prcp_roll30"] = prcp_roll30

            # Drop time if present; keep exactly feature columns
            if "time" in feat.index:
                feat = feat.drop(labels=["time"])

            X = pd.DataFrame([feat.to_dict()])

            pred_tmax = float(reg_pipe.predict(X)[0])
            proba_rain = float(clf_pipe.predict_proba(X)[:, 1][0])
            pred_rain = int(proba_rain >= 0.5)

            results.append({
                "city": city,
                "observed_asof_date": observed_asof,
                "forecast_horizon_day": step,
                "predicted_for_date": predicted_for.date(),
                "pred_tmax": pred_tmax,
                "pred_rain": pred_rain,
                "proba_rain": proba_rain
            })

            # ---- Update state for next step (iterative) ----
            # Update tmax lags: shift them forward
            tmax_lag3 = tmax_lag2
            tmax_lag2 = tmax_lag1
            tmax_lag1 = last_tmax
            last_tmax = pred_tmax  # we use predicted tmax as next day's "last known"

            # For rain features, we don't predict next-day prcp amount; keep last known prcp
            # If you want, we can add a prcp regression model later.
            prcp_lag1 = last_prcp

            # Rolling approximations (simple exponential-ish smoothing)
            if not np.isnan(tmax_roll7):
                tmax_roll7 = 0.85 * tmax_roll7 + 0.15 * last_tmax
            if not np.isnan(tmax_roll30):
                tmax_roll30 = 0.95 * tmax_roll30 + 0.05 * last_tmax

            if not np.isnan(prcp_roll7):
                prcp_roll7 = 0.85 * prcp_roll7 + 0.15 * last_prcp
            if not np.isnan(prcp_roll30):
                prcp_roll30 = 0.95 * prcp_roll30 + 0.05 * last_prcp

    out = pd.DataFrame(results).sort_values(["city", "predicted_for_date"])
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
    print(out.head(20))

if __name__ == "__main__":
    main()
