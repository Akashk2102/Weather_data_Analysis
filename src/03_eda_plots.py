import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/features_daily.csv"
PLOTS_DIR = "outputs/plots"

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run src/02_prepare_features.py first.")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    sns.set_style("whitegrid")

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])

    monthly = (
        df.groupby(["city", df["time"].dt.to_period("M")])[["tmax", "prcp"]]
          .mean()
          .reset_index()
    )
    monthly["time"] = monthly["time"].dt.to_timestamp()

    plt.figure(figsize=(11, 5))
    sns.lineplot(data=monthly, x="time", y="tmax", hue="city")
    plt.title("Monthly Avg Tmax (°C) by City")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/monthly_tmax.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 5))
    sns.lineplot(data=monthly, x="time", y="prcp", hue="city")
    plt.title("Monthly Avg Precipitation (mm) by City")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/monthly_prcp.png", dpi=200)
    plt.close()

    num = df.select_dtypes("number")
    corr = num.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png", dpi=200)
    plt.close()

    print(f"Saved plots -> {PLOTS_DIR}")

if __name__ == "__main__":
    main()
