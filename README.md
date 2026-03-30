
Markdown
# Weather Data Analysis (Data Science Mini Project)

This project is an end-to-end **data science pipeline** for daily weather forecasting. It:
1. Fetches daily weather data for multiple cities
2. Performs feature engineering (lags, rolling averages, calendar features)
3. Trains machine learning models for:
   - **Regression:** predict **tomorrow’s max temperature** (`tmax_tomorrow`)
   - **Classification:** predict **whether it will rain tomorrow** (`rain_tomorrow`)
4. Evaluates models and generates plots (feature importance, ROC curve, confusion matrix)
5. Produces **next-day predictions** and an optional **next 7-day iterative forecast**

> Important: Predictions are generated for the **next day after the latest available date** in the dataset (`observed_asof_date + 1`).  
> If the data provider is not fully up-to-date, the predicted date may be earlier than today.

---

## Cities used
Configured in `src/01_fetch_data.py`:
- Bengaluru
- Chikkaballapura
- Ramanagara

---

## What the project does (Data Science)
- Converts raw daily weather time-series data into a supervised ML dataset by creating **tomorrow targets**.
- Uses time-series feature engineering:
  - **Lag features** (previous days): `tmax_lag1`, `prcp_lag1`, etc.
  - **Rolling window features**: `tmax_roll7`, `prcp_roll30`, etc.
  - **Seasonality features**: `month`, `dayofyear`, `year`
- Trains and evaluates models with standard DS metrics:
  - Regression: MAE, RMSE, R²
  - Classification: Accuracy, F1-score, ROC-AUC

---

## Project structure
- `src/`
  - `01_fetch_data.py` — fetch/update daily weather data into `data/raw_daily.csv`
  - `02_prepare_features.py` — generate ML dataset `data/features_daily.csv` and `data/predict_today_input.csv`
  - `03_eda_plots.py` — EDA plots (monthly trends, correlation heatmap)
  - `04_train_models.py` — trains models and saves them to `outputs/models/`
  - `05_feature_importance.py` — feature importance plots for both models
  - `06_eval_plots.py` — evaluation plots (ROC curve, confusion matrix, regression scatter)
  - `07_export_predictions.py` — exports test-set predictions to CSV
  - `08_predict_next_day.py` — predicts **next day after latest available date** (one row per city)
  - `09_forecast_next_7_days.py` — iterative next 7-day forecast (7 rows per city)

- `data/`
  - `raw_daily.csv` — fetched raw daily weather data
  - `features_daily.csv` — engineered dataset used for training
  - `predict_today_input.csv` — latest row per city used as input for prediction scripts

- `outputs/`
  - `metrics/metrics.json` — saved model evaluation metrics
  - `models/` — saved ML models (joblib)
  - `plots/` — generated plots
  - `predictions/` — generated prediction CSV files

---

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Run the full pipeline
PowerShell
.\.venv\Scripts\python.exe src\01_fetch_data.py
.\.venv\Scripts\python.exe src\02_prepare_features.py
.\.venv\Scripts\python.exe src\03_eda_plots.py
.\.venv\Scripts\python.exe src\04_train_models.py
.\.venv\Scripts\python.exe src\05_feature_importance.py
.\.venv\Scripts\python.exe src\06_eval_plots.py
.\.venv\Scripts\python.exe src\08_predict_next_day.py
Optional: export test-set predictions
PowerShell
.\.venv\Scripts\python.exe src\07_export_predictions.py
Optional: iterative next 7-day forecast
PowerShell
.\.venv\Scripts\python.exe src\09_forecast_next_7_days.py
Outputs generated
outputs/metrics/metrics.json
outputs/plots/ (examples)
feature_importance_regression.png
feature_importance_classification.png
clf_roc_curve.png
clf_confusion_matrix.png
outputs/predictions/
predictions_next_day.csv
forecast_next_7_days.csv (if you run the 7-day script)
Notes / Limitations
Weather data freshness depends on the data provider (station availability + update frequency).
The 7-day forecast is iterative (future days use previous predictions), so uncertainty increases with horizon.
