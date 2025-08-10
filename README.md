# Wind Turbine Gearbox Temperature Prediction System

Early warning system for wind-turbine gearbox overheating using deep learning. It forecasts **48 hours** of gearbox temperature at **15‑minute** resolution and supports two ways to use it:

- **Streamlit dashboard** for interactive, batch predictions and visualization.
- **FastAPI** inference service for programmatic access (Postman, Swagger UI, curl, or any client).

---

## 📂 Current Project Layout (matches your repo)

```
.
├─ 2_day_gearbox.ipynb                      # Training / experimentation notebook
├─ inference.ipynb                          # Ad‑hoc inference tests (not required to run API)
├─ api.ipynb                                # Optional: API prototyping notebook
├─ app.py                                   # Streamlit dashboard
├─ fastapi_app.py                           # FastAPI server (production-ish)
├─ wind_infer.py                            # WindTurbineInference class (used by API)
├─ wt84_with_alarms.csv                     # Raw data
├─ test_data_processed.csv                  # Processed test set (optional helper for app)
├─ improved_model.h5                        # Trained model
├─ feature_scaler.pkl                       # Feature scaler
├─ target_scaler.pkl                        # Target scaler
├─ common_features.json                     # Feature names (order must match scaler)
├─ label_encoder_alarm_system.pkl           # Label encoder for alarm_system
├─ actual_last_48h.csv                      # Example export: last 48h actuals
├─ pred_next_48h.csv                        # Example export: next 48h predictions
├─ actual_pred_combined.csv                 # Example export: merged view
├─ requirements.txt                         # Python dependencies
├─ README.md                                # This file
└─ Wind Turbine Gearbox Temperature Prediction Report.docx  # Project report (optional)
```
*(You may also have a virtual‑env folder like `gearbox-lstm-env/`; it’s not required to be inside the repo.)*

---

## 🔄 Workflow Overview

### 1) Train (not required if model files already exist)
Run **`2_day_gearbox.ipynb`** to:
- Clean & resample data to 15‑minute cadence.
- Engineer features (lags, deltas, rolling means, cyclical time, alarm features).
- Train **Conv1D + BiLSTM** model.
- Export artifacts: `improved_model.h5`, `feature_scaler.pkl`, `target_scaler.pkl`, `common_features.json`, and (optionally) `label_encoder_alarm_system.pkl`.

### 2) Deploy one (or both) of:
- **Streamlit Dashboard** (`app.py`) – interactive, visual.
- **FastAPI Service** (`fastapi_app.py`) – HTTP JSON API for integration/testing.

---

## 🚀 Quick Start

### A. Set up environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
```

> **TensorFlow note:** If install fails on your machine, pin a CPU build, e.g. `pip install tensorflow==2.10.0`.

---

### B. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open: http://localhost:8501

**What you can do**
- Select batches (e.g., 10–200 sequences) and run predictions.
- See 48‑h prediction curves vs actuals.
- Apply the **65 °C Warning** and **70 °C Failure** thresholds.
- Export CSVs (e.g., `pred_next_48h.csv`, `actual_last_48h.csv`, `actual_pred_combined.csv`).

---

### C. Run the FastAPI Inference Service
```bash
uvicorn fastapi_app:app --reload --port 8000
```
- Health check: http://127.0.0.1:8000/health  
- Swagger UI (interactive, no extra tools needed): http://127.0.0.1:8000/docs  
- ReDoc: http://127.0.0.1:8000/redoc

**POST /predict** (JSON body)
```json
{
  "timestamp": "2014-11-15 06:30:00",
  "critical_temp": 65.0
}
```
Allowed optional overrides (use only if you know what you’re doing):
```json
{
  "timestamp": "2014-11-15 06:30:00",
  "lookback_steps": 384,   // multiple of 4 (15-min steps)
  "forecast_steps": 192,   // multiple of 4
  "critical_temp": 65.0
}
```

**curl example**
```bash
curl -X POST "http://127.0.0.1:8000/predict"      -H "Content-Type: application/json"      -d '{"timestamp":"2014-11-15 06:30:00","critical_temp":65.0}'
```

**What the API returns**
- `prediction_start`, `prediction_end` (48‑h window)
- `exceeded`, `first_exceed_time`, `max_temperature`, `max_temperature_time`
- `critical_temperature_threshold`, `total_exceed_count`
- `overlap_metrics` (MAE / RMSE / Bias where actuals exist)
- `predictions`: 192 points (`timestamp`, `predicted_temperature`, and if available `actual_temperature`, `abs_error`).

**How it works under the hood**
- Aligns your timestamp to the 15‑min grid.
- Builds a **lookback window** of **96 hours** (default `lookback_steps=384`) ending at that time.
- Predicts **48 hours** ahead (default `forecast_steps=192`) at 15‑min resolution.
- Optionally merges **actuals** from `wt84_with_alarms.csv` for error metrics.

> Tip: If you pass `lookback_steps=0` or `forecast_steps=0`, the API will reject the request (input validators). Omit them to use defaults.

---

## 🎯 Model & Thresholds

- **Architecture**: Conv1D → BiLSTM → Dense
- **Lookback window**: 96h (384 steps @ 15‑min)
- **Forecast horizon**: 48h (192 steps @ 15‑min)
- **Warning threshold**: 65 °C
- **Failure threshold**: 70 °C

### Performance (example on your latest run)
| Metric    | Training | Validation | Test  |
|----------:|---------:|-----------:|------:|
| MAE (°C)  | 5.385    | 8.153      | 5.911 |
| RMSE (°C) | 8.506    | 10.169     | 8.163 |
| R²        | 0.533    | 0.680      | 0.738 |

Failure detection (sample setup):
- **Precision** ≈ 80%
- **Recall** ≈ 58%
- **Accuracy** ≈ 80.86%

> These will vary by data window; keep thresholds configurable via the API.

---

## 🧪 Reproducing the Side‑by‑Side View (Actual vs Predicted)

The API uses `WindTurbineInference.predict_from_point(...)` (see `wind_infer.py`), which:
1. Trims history up to the chosen time and fills a continuous 5‑min grid.
2. Resamples to 15‑min, engineers features (lags, deltas, rollings, cyclic time, alarm features).
3. Builds the last 96h lookback ending exactly at the source time (padded if short).
4. Scales features with saved `feature_scaler.pkl` and predicts 48h ahead.
5. Inverse‑scales the outputs and **merges actuals** (nearest‑match tolerance 7m30s) to compute MAE/RMSE/Bias.
6. Returns a DataFrame with `timestamp`, `predicted_temperature`, `actual_temperature` (if available), and `abs_error`.

---

## 🐛 Troubleshooting

- **`Timestamp not found`** in `/predict`  
  Make sure the time exists (or within ±5 min). The API picks the nearest row within 5 minutes.

- **TensorFlow install issues**  
  Try: `pip install tensorflow==2.10.0` (CPU). On Apple Silicon or Windows, follow platform‑specific TF install guides.

- **Feature order mismatch**  
  Ensure `common_features.json` matches `feature_scaler.feature_names_in_`. Re‑export if you re‑train.

- **Memory/slow predictions**  
  Use CPU‑only TF if GPU isn’t available; keep a single `Uvicorn` worker for local testing.

- **CORS for web apps**  
  For a browser client from a different origin, add FastAPI CORS middleware. Postman/Swagger UI won’t need it locally.

---

## 🧱 Requirements

Install via `pip install -r requirements.txt`. If you need to regenerate it:
```
fastapi
uvicorn
pydantic
pandas
numpy
scikit-learn
tensorflow
joblib
streamlit
matplotlib
seaborn
jupyter
```

---

## 📌 Notes

- Keep **`lookback_steps` = 384** and **`forecast_steps` = 192** unless you retrain.
- Keep all artifacts in the same folder as `fastapi_app.py` for simplest paths.
- The report `.docx` is informational—API/app don’t depend on it.
- The CSV exports (`actual_last_48h.csv`, `pred_next_48h.csv`, `actual_pred_combined.csv`) are examples produced by the dashboard; they are not required for the API.

---

## ✨ Roadmap

- Weather inputs
- Multi‑turbine fleet view
- Scheduled retraining
- Streaming ingestion & online inference
- Mobile‑friendly UI

---

**Happy forecasting!** ⚡🧠🌬️
