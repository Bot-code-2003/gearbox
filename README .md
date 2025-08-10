# Wind Turbine Gearbox Temperature Prediction System

Early warning system for wind-turbine gearbox overheating using deep learning. It forecasts **48 hours** of gearbox temperature at **15‑minute** resolution and supports two ways to use it:

- **Streamlit dashboard** for interactive, batch predictions and visualization.
- **FastAPI** inference service for programmatic access (Postman, Swagger UI, curl, or any client).

---

## 📂 Current Project Layout

```
.
├─ 2_day_gearbox.ipynb                      # Training / experimentation notebook
├─ inference.ipynb                          # Ad‑hoc inference tests (not required to run API)
├─ app.py                                   # Streamlit dashboard
├─ fastapi_app.py                           # FastAPI server
├─ wind_infer.py                            # WindTurbineInference class (used by API)
├─ wt84_with_alarms.csv                     # Raw data
├─ test_data_processed.csv                  # Processed test set
├─ improved_model.h5                        # Trained model
├─ feature_scaler.pkl                       # Feature scaler
├─ target_scaler.pkl                        # Target scaler
├─ common_features.json                     # Feature names
├─ label_encoder_alarm_system.pkl           # Label encoder for alarm_system
├─ sample_row.json                          # Example JSON row
├─ requirements.txt                         # Python dependencies
├─ README.md                                # This file
└─ Wind Turbine Gearbox Temperature Prediction Report.docx
```

---

## 🔄 Workflow Overview

### 1) Train (optional)

Use **`2_day_gearbox.ipynb`** to train and export model artifacts.

### 2) Deploy options

- **Streamlit Dashboard** (`app.py`)
- **FastAPI Service** (`fastapi_app.py`)

---

## 🚀 Quick Start

### A. Environment Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

> If TensorFlow fails to install, try:  
> `pip install tensorflow==2.10.0`

---

### B. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

Open: http://localhost:8501

---

### C. Run the FastAPI Service

```bash
uvicorn fastapi_app:app --reload --port 8000
```

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- Health check: http://127.0.0.1:8000/health

---

## ✅ Checking if the API Works

1. **Open Swagger UI**  
   Run:

   ```bash
   uvicorn fastapi_app:app --reload --port 8000
   ```

   Then open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

2. **Test `GET /sample-row/{timestamp}`**  
   Use:

   ```
   2012-04-06 01:05:00
   ```

   This will return a JSON row from `wt84_with_alarms.csv`.  
   Example output:

   ```json
   {
     "date_time": "2012-04-06 01:05:00",
     "wgen_avg_Spd": 8.5,
     "wgdc_avg_TriGri_PwrAt": 1200.0,
     "wtrm_avg_TrmTmp_Gbx": 45.2,
     "wtrm_avg_TrmTmp_GbxBrg151": 42.1,
     "wtrm_avg_TrmTmp_GbxBrg452": 43.8,
     "wtrm_avg_Gbx_OilPres": 5.2,
     "alarm_system": "Normal",
     "alarm_desc": "No alarm"
   }
   ```

3. **Test `POST /predict`**
   - Copy the JSON output from the `GET /sample-row` response.
   - Paste it into the `row_data` field in `/predict` request body in Swagger UI.
   - Click **Execute**.  
     You should receive prediction results with 48-hour forecast.

---

## 🎯 Model & Thresholds

- **Architecture**: Conv1D → BiLSTM → Dense
- **Lookback**: 384 steps (96h)
- **Forecast**: 192 steps (48h)
- **Warning**: 65 °C
- **Failure**: 70 °C

---

## 🧱 Requirements

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

**Happy forecasting!** ⚡🧠🌬️
