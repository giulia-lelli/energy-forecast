import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from features import FEATURE_COLS, build_features, load_and_clean

# --- App setup ---
app = FastAPI(
    title="Energy Demand Forecast API",
    description="Predicts hourly energy consumption in MW for the PJM grid",
    version="1.0.0"
)

# --- Load model and historical data at startup ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm_energy.txt")
DATA_PATH  = os.path.join(BASE_DIR, "data", "raw", "PJM_Load_hourly.csv")

model = lgb.Booster(model_file=MODEL_PATH)
df_history = load_and_clean(DATA_PATH)
df_history = build_features(df_history)


# --- Request / Response schemas ---
class ForecastRequest(BaseModel):
    datetime: str  # Format: "YYYY-MM-DD HH:00:00"

class ForecastResponse(BaseModel):
    datetime: str
    predicted_mw: float
    unit: str = "MW"


# --- Endpoints ---
@app.get("/")
def root():
    return {
        "message": "Energy Forecast API is running",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    try:
        dt = pd.to_datetime(request.datetime)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid datetime format. Use YYYY-MM-DD HH:00:00")

    # Find the row in history matching this datetime
    row = df_history[df_history['Datetime'] == dt]

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data found for {dt}. Available range: {df_history['Datetime'].min()} to {df_history['Datetime'].max()}"
        )

    features = row[FEATURE_COLS].values
    prediction = model.predict(features)[0]

    return ForecastResponse(
        datetime=str(dt),
        predicted_mw=round(float(prediction), 2)
    )

@app.get("/forecast/{datetime_str}")
def forecast_get(datetime_str: str):
    return forecast(ForecastRequest(datetime=datetime_str.replace("T", " ")))
