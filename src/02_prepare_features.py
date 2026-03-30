import os
import numpy as np
import pandas as pd

RAW_PATH = "data/raw_daily.csv"
FEATURES_OUT = "data/features_daily.csv"
PREDICT_TODAY_OUT = "data/predict_today_input.csv"

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["dayofyear"] = df["time"].dt.dayofyear
    return df

def add_lag_roll(df: pd.DataFrame, col: str, lags=(1, 2, 3), rolls=(7, 30)) -> pd.DataFrame:
    for lag in lags:
        df[f"{col}_lag{lag}"] = df.groupby("city")[col].shift(lag)
    for w in rolls:
        df[f"{col}_roll{w}"] = (
            df.groupby("city")[col]
              .shift(1)
              .rolling(window=w)
              .mean()
              .reset_index(level=0, drop=True)
        )
    return df

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Missing {RAW_PATH}. Run src/01_fetch_data.py first.")

    df = pd.read_csv(RAW_PATH)
    df = df.dropna(subset=["time", "city"]).copy()
    df = add_time_features(df)
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    # Basic sanity cleaning
    if "tmax" in df.columns:
        df.loc[(df["tmax"] < -10) | (df["tmax"] > 60), "tmax"] = np.nan
    if "tmin" in df.columns:
        df.loc[(df["tmin"] < -20) | (df["tmin"] > 50), "tmin"] = np.nan
    if "prcp" in df.columns:
        df.loc[df["prcp"] < 0, "prcp"] = np.nan

    if "tmax" not in df.columns or "prcp" not in df.columns:
        raise ValueError("Expected columns 'tmax' and 'prcp' not found in raw data.")

    # Targets for training (tomorrow relative to each row)
    df["tmax_tomorrow"] = df.groupby("city")["tmax"].shift(-1)
    df["prcp_tomorrow"] = df.groupby("city")["prcp"].shift(-1)
    df["rain_tomorrow"] = (df["prcp_tomorrow"].fillna(0) > 0).astype(int)

    # Feature engineering
    for col in ["tmax", "tmin", "tavg", "prcp", "wspd", "pres"]:
        if col in df.columns:
            df = add_lag_roll(df, col)

    # Training dataset: rows where tomorrow targets exist
    train_df = df.dropna(subset=["tmax_tomorrow", "prcp_tomorrow"]).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    train_df.to_csv(FEATURES_OUT, index=False)
    print(f"Saved -> {FEATURES_OUT}")
    print("Train shape:", train_df.shape)

    # Prediction input for TODAY: use the latest available row per city (which should be yesterday)
    latest_rows = (
        df.sort_values(["city", "time"])
          .groupby("city", as_index=False)
          .tail(1)
          .copy()
    )

    # For predicting today's outcomes, we should NOT include target columns
    # Keep same columns as training features at prediction time
    drop_cols = ["tmax_tomorrow", "prcp_tomorrow", "rain_tomorrow"]
    pred_input = latest_rows.drop(columns=[c for c in drop_cols if c in latest_rows.columns])

    pred_input.to_csv(PREDICT_TODAY_OUT, index=False)
    print(f"Saved -> {PREDICT_TODAY_OUT}")
    print("Predict rows (1 per city):", len(pred_input))
    print(pred_input[["city","time"]])


if __name__ == "__main__":
    main()
