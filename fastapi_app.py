# fastapi_app.py
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
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
    timestamp: str = Field(..., example="2014-11-15 06:30:00")
    lookback_steps: Optional[int] = None
    forecast_steps: Optional[int] = None
    critical_temp: Optional[float] = None

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

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # parse and find the row for this timestamp (or nearest within 5 minutes)
    try:
        ts = pd.to_datetime(req.timestamp)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timestamp format. Use 'YYYY-MM-DD HH:MM:SS'.")

    row = history_df.loc[history_df["date_time"] == ts]
    if row.empty:
        nearest = history_df.iloc[(history_df["date_time"] - ts).abs().argsort()[:1]]
        if nearest.empty or abs(nearest.iloc[0]["date_time"] - ts) > pd.Timedelta(minutes=5):
            raise HTTPException(status_code=404, detail="Timestamp not found in history.")
        row = nearest

    # optional overrides
    if req.lookback_steps is not None:
        pipe.lookback_steps = int(req.lookback_steps)
    if req.forecast_steps is not None:
        pipe.forecast_steps = int(req.forecast_steps)
    if req.critical_temp is not None:
        pipe.critical_temp = float(req.critical_temp)

    latest_point = row.iloc[0].to_dict()

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
    preds_records = []
    for _, r in preds.iterrows():
        preds_records.append(PredictionPoint(
            timestamp=r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            predicted_temperature=float(r["predicted_temperature"]),
            actual_temperature=(None if pd.isna(r.get("actual_temperature")) else float(r["actual_temperature"])),
            abs_error=(None if pd.isna(r.get("abs_error")) else float(r["abs_error"]))
        ))

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
