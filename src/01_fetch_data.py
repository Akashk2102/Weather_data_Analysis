from datetime import datetime, timedelta
import os
import pandas as pd
from meteostat import Daily, Stations

HIST_START = datetime(2015, 1, 1)
END = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

CITIES = [
    ("Bengaluru",        12.9716, 77.5946),
    ("Ramanagara",       12.7206, 77.2800),
    ("Chikkaballapura",  13.4355, 77.7315),
]

OUT_PATH = "data/raw_daily.csv"

def best_station(lat: float, lon: float):
    # Find nearby stations and pick the first (closest) with daily inventory
    st = Stations().nearby(lat, lon).inventory("daily", (HIST_START, END)).fetch(10)
    if st is None or len(st) == 0:
        return None
    # Sort by distance if present, else just take first row
    if "distance" in st.columns:
        st = st.sort_values("distance")
    return st.index[0]  # station id

def fetch_city_daily(name: str, lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    if start > end:
        return pd.DataFrame()

    station_id = best_station(lat, lon)
    if station_id is None:
        print(f"[{name}] No station found with daily inventory")
        return pd.DataFrame()

    df = Daily(station_id, start, end).fetch()
    if df is None or len(df) == 0:
        print(f"[{name}] No data returned for station {station_id}")
        return pd.DataFrame()

    df = df.reset_index()
    df["city"] = name
    df["station_id"] = station_id
    return df

def main():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(OUT_PATH):
        existing = pd.read_csv(OUT_PATH)
        existing["time"] = pd.to_datetime(existing["time"])
    else:
        existing = pd.DataFrame(columns=["time", "city"])

    new_frames = []

    for (name, lat, lon) in CITIES:
        city_existing = existing[existing["city"] == name] if len(existing) else pd.DataFrame()

        if len(city_existing) > 0:
            last_date = pd.to_datetime(city_existing["time"]).max()
            start = (last_date + timedelta(days=1)).to_pydatetime()
        else:
            start = HIST_START

        print(f"[{name}] fetching {start.date()} -> {END.date()}")
        df_new = fetch_city_daily(name, lat, lon, start, END)
        if len(df_new) > 0:
            new_frames.append(df_new)

    combined = existing.copy()
    if len(new_frames) > 0:
        combined = pd.concat([existing, pd.concat(new_frames, ignore_index=True)], ignore_index=True)

    keep = ["time", "city", "station_id", "tavg", "tmin", "tmax", "prcp", "wspd", "pres"]
    keep = [c for c in keep if c in combined.columns]
    combined = combined[keep]

    combined["time"] = pd.to_datetime(combined["time"])
    combined = combined.drop_duplicates(subset=["city", "time"]).sort_values(["city", "time"]).reset_index(drop=True)

    combined.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
    print("Cities:", combined["city"].unique())
    print("Date range:", combined["time"].min().date(), "to", combined["time"].max().date())

if __name__ == "__main__":
    main()
