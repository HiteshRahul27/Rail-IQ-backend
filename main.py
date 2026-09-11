"""
Rail IQ prediction backend.

  GET /trains
      List every train the model knows about, with its station list.

  GET /predict-eta?train_id=12805&current_station_no=10&current_delay=12&as_of=2026-09-10T14:00:00
      Predicts delay (and ETA) at every remaining station on the route,
      station by station, feeding each prediction back in as the next
      station's "current delay" input — this is model INFERENCE, not
      retraining, matching Step 2.4 of the brief.

Run locally:
    uvicorn main:app --reload --port 8000

Deploy: see README.md (Render, free tier).
"""
import json
from datetime import datetime, timedelta
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Rail IQ Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo scope — tighten to your Vercel domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
train_encoder = joblib.load("train_encoder.pkl")
with open("data/stations_reference.json") as f:
    STATIONS_REF = json.load(f)
with open("model_metrics.json") as f:
    MODEL_METRICS = json.load(f)

FEATURES = ["prev_delay", "distance_to_next_km", "hour_of_day", "dow", "stations_from_origin", "train_id_enc"]


class StationPrediction(BaseModel):
    station_no: int
    code: str
    name: str
    distance_km: int
    scheduled_offset_min: float
    predicted_delay_min: float
    predicted_eta: str


class PredictResponse(BaseModel):
    train_id: str
    train_name: str
    current_station_no: int
    current_delay_min: float
    predictions: list[StationPrediction]
    model_metrics: dict


@app.get("/")
def root():
    return {"service": "rail-iq-prediction-api", "status": "ok", "model": MODEL_METRICS["model_kind"]}


@app.get("/trains")
def list_trains():
    return [
        {"train_id": tid, "train_name": v["train_name"], "stations": v["stations"]}
        for tid, v in STATIONS_REF.items()
    ]


@app.get("/predict-eta", response_model=PredictResponse)
def predict_eta(
    train_id: str = Query(..., description="e.g. 12805"),
    current_station_no: int = Query(1, description="Station index the train has just left/reached"),
    current_delay: float = Query(0, description="Known delay (min) at current_station_no"),
    as_of: Optional[str] = Query(None, description="ISO timestamp for time-of-day features; defaults to now"),
):
    if train_id not in STATIONS_REF:
        raise HTTPException(404, f"Unknown train_id '{train_id}'. Try one of {list(STATIONS_REF.keys())}")

    ref = STATIONS_REF[train_id]
    stations = ref["stations"]
    train_id_enc = int(train_encoder.transform([train_id])[0])
    now = datetime.fromisoformat(as_of) if as_of else datetime.utcnow()

    remaining = [s for s in stations if s["station_no"] > current_station_no]
    if not remaining:
        raise HTTPException(400, "current_station_no is already the last station on this route")

    prev_station = next((s for s in stations if s["station_no"] == current_station_no), stations[0])
    prev_delay = current_delay
    prev_distance = prev_station["distance_km"]

    predictions = []
    for s in remaining:
        row = pd.DataFrame([{
            "prev_delay": prev_delay,
            "distance_to_next_km": max(0, s["distance_km"] - prev_distance),
            "hour_of_day": s["scheduled_hour"],
            "dow": now.weekday(),
            "stations_from_origin": s["station_no"],
            "train_id_enc": train_id_enc,
        }])[FEATURES]
        pred = float(model.predict(row)[0])
        pred = max(0.0, round(pred, 1))

        eta = now + timedelta(minutes=pred)
        predictions.append(StationPrediction(
            station_no=s["station_no"], code=s["code"], name=s["name"],
            distance_km=s["distance_km"], scheduled_offset_min=s["scheduled_offset_min"],
            predicted_delay_min=pred, predicted_eta=eta.strftime("%Y-%m-%d %H:%M"),
        ))
        prev_delay = pred
        prev_distance = s["distance_km"]

    return PredictResponse(
        train_id=train_id, train_name=ref["train_name"],
        current_station_no=current_station_no, current_delay_min=current_delay,
        predictions=predictions, model_metrics=MODEL_METRICS,
    )
