
# fastapi_app.py
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from wind_infer import WindTurbineInference  # <-- uses your class

# ---- Config paths (env or defaults) ----
CSV_PATH = os.getenv("WT_HISTORY_CSV", "wt84_with_alarms.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "improved_model.h5")
FEATURE_SCALER_PATH = os.getenv("FEATURE_SCALER_PATH", "feature_scaler.pkl")
TARGET_SCALER_PATH = os.getenv("TARGET_SCALER_PATH", "target_scaler.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "common_features.json")
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", "label_encoder_alarm_system.pkl")

LOOKBACK_STEPS_DEFAULT = 384  # keep same as training (4 days @ 15 min)
FORECAST_STEPS_DEFAULT = 192  # 2 days @ 15 min
CRITICAL_TEMP_DEFAULT = 65.0

app = FastAPI(title="WindTurbine Forecast API")

# ---- Load once on startup ----
history_df: pd.DataFrame = None
pipe: WindTurbineInference = None

@app.on_event("startup")
def _startup():
    global history_df, pipe
    history_df = pd.read_csv(CSV_PATH, parse_dates=["date_time"])
    pipe = WindTurbineInference(
        model_path=MODEL_PATH,
        feature_scaler_path=FEATURE_SCALER_PATH,
        target_scaler_path=TARGET_SCALER_PATH,
        features_path=FEATURES_PATH,
        label_encoder_path=LABEL_ENCODER_PATH,
        lookback_steps=LOOKBACK_STEPS_DEFAULT,
        forecast_steps=FORECAST_STEPS_DEFAULT,
        critical_temp=CRITICAL_TEMP_DEFAULT
    )

class PredictRequest(BaseModel):
    row_data: Dict[str, Any] = Field(..., description="Complete row from raw dataset as JSON")

    class Config:
        schema_extra = {
            "example": {
                "row_data": {
                    "date_time": "2014-11-15 06:30:00",
                    "wgen_avg_Spd": 8.5,
                    "wgdc_avg_TriGri_PwrAt": 1200.0,
                    "wtrm_avg_TrmTmp_Gbx": 45.2,
                    "wtrm_avg_TrmTmp_GbxBrg151": 42.1,
                    "wtrm_avg_TrmTmp_GbxBrg452": 43.8,
                    "wtrm_avg_Gbx_OilPres": 5.2,
                    "alarm_system": "Normal",
                    "alarm_desc": "No alarm"
                }
            }
        }


class PredictionPoint(BaseModel):
    timestamp: str
    predicted_temperature: float
    actual_temperature: Optional[float] = None
    abs_error: Optional[float] = None

class PredictResponse(BaseModel):
    prediction_start: str
    prediction_end: str
    exceeded: bool
    first_exceed_time: Optional[str] = None
    max_temperature: float
    max_temperature_time: str
    critical_temperature_threshold: float
    total_exceed_count: int
    overlap_metrics: dict
    predictions: List[PredictionPoint]

@app.get("/health")
def health():
    return {"status": "ok", "rows_in_history": int(len(history_df))}

@app.get("/sample-row/{timestamp}")
def get_sample_row(timestamp: str):
    """
    Get a single row from the dataset for the given timestamp.
    """
    try:
        ts = pd.to_datetime(timestamp)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid timestamp format. Use 'YYYY-MM-DD HH:MM:SS'."
        )

    row = history_df.loc[history_df["date_time"] == ts]
    if row.empty:
        nearest = history_df.iloc[(history_df["date_time"] - ts).abs().argsort()[:1]]
        if nearest.empty or abs(nearest.iloc[0]["date_time"] - ts) > pd.Timedelta(minutes=5):
            raise HTTPException(status_code=404, detail="Timestamp not found in history.")
        row = nearest

    row_dict = row.iloc[0].to_dict()

    # Convert timestamps & NaN
    for key, value in row_dict.items():
        if isinstance(value, pd.Timestamp):
            row_dict[key] = value.strftime("%Y-%m-%d %H:%M:%S")
        elif pd.isna(value):
            row_dict[key] = None

    return row_dict


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Validate that row_data contains date_time
    if "date_time" not in req.row_data:
        raise HTTPException(status_code=400, detail="row_data must contain 'date_time' field.")

    # Parse timestamp
    try:
        ts = pd.to_datetime(req.row_data["date_time"])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date_time format in row_data. Use 'YYYY-MM-DD HH:MM:SS'.")

    # Prepare latest_point
    latest_point = req.row_data.copy()
    latest_point["date_time"] = ts

    # Always use defaults
    pipe.lookback_steps = LOOKBACK_STEPS_DEFAULT
    pipe.forecast_steps = FORECAST_STEPS_DEFAULT
    pipe.critical_temp = CRITICAL_TEMP_DEFAULT

    try:
        out = pipe.predict_from_point(
            latest_point=latest_point,
            history=history_df,
            actual_df=history_df
        )
    except AssertionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    preds = out["predictions"]
    preds_records = [
        PredictionPoint(
            timestamp=r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            predicted_temperature=float(r["predicted_temperature"]),
            actual_temperature=(None if pd.isna(r.get("actual_temperature")) else float(r["actual_temperature"])),
            abs_error=(None if pd.isna(r.get("abs_error")) else float(r["abs_error"]))
        )
        for _, r in preds.iterrows()
    ]

    return PredictResponse(
        prediction_start=str(out["prediction_start"]),
        prediction_end=str(out["prediction_end"]),
        exceeded=bool(out["exceeded"]),
        first_exceed_time=(None if out["first_exceed_time"] is None else str(out["first_exceed_time"])),
        max_temperature=float(out["max_temperature"]),
        max_temperature_time=str(out["max_temperature_time"]),
        critical_temperature_threshold=float(out["critical_temperature_threshold"]),
        total_exceed_count=int(out["total_exceed_count"]),
        overlap_metrics=out.get("overlap_metrics", {}),
        predictions=preds_records
    )


@app.get("/dataset-info")
def dataset_info():
    """Get basic information about the loaded dataset"""
    return {
        "total_rows": len(history_df),
        "date_range": {
            "start": history_df["date_time"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "end": history_df["date_time"].max().strftime("%Y-%m-%d %H:%M:%S")
        },
        "columns": list(history_df.columns),
        "sample_timestamps": [
            ts.strftime("%Y-%m-%d %H:%M:%S") 
            for ts in history_df["date_time"].head(10)
        ]
    }