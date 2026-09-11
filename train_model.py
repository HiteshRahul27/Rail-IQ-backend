"""
Step 2: Build a real (simple) prediction model.

Features per row (= arrival at a given station):
  - current_delay: delay (min) recorded at the PREVIOUS station on this journey
  - distance_to_next_km: distance (km) covered on the leg into this station
  - hour_of_day: scheduled hour of arrival at this station (0-23)
  - dow: day of week of the journey (0=Mon)
  - stations_from_origin: how many stops into the journey this is
  - train_id: which train (label-encoded)

Target:
  - delay_minutes: the actual recorded delay at this station

Split: 80/20 by (train_id, journey_date) — whole journeys go to one side,
so the model is evaluated on journeys it never saw, not just rows.
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from lightgbm import LGBMRegressor
    MODEL_KIND = "lightgbm"
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as LGBMRegressor
    MODEL_KIND = "sklearn-gbm"

df = pd.read_csv("data/training_data.csv", parse_dates=["scheduled_time", "actual_time"], dtype={"train_id": str, "station_code": str})
df = df.dropna(subset=["delay_minutes"]).reset_index(drop=True)
df = df.sort_values(["train_id", "journey_date", "station_no"]).reset_index(drop=True)

# --- feature engineering -----------------------------------------------
df["prev_delay"] = df.groupby(["train_id", "journey_date"])["delay_minutes"].shift(1).fillna(0)
df["prev_distance_km"] = df.groupby(["train_id", "journey_date"])["distance_km"].shift(1).fillna(0)
df["distance_to_next_km"] = (df["distance_km"] - df["prev_distance_km"]).clip(lower=0)
df["hour_of_day"] = df["scheduled_time"].dt.hour
df["dow"] = df["scheduled_time"].dt.dayofweek
df["stations_from_origin"] = df["station_no"]

train_encoder = LabelEncoder()
df["train_id_enc"] = train_encoder.fit_transform(df["train_id"])

FEATURES = ["prev_delay", "distance_to_next_km", "hour_of_day", "dow", "stations_from_origin", "train_id_enc"]
TARGET = "delay_minutes"

# --- group-aware 80/20 split (whole journeys, not rows) ----------------
df["journey_group"] = df["train_id"] + "_" + df["journey_date"]
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["journey_group"]))
train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

model = LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1) \
    if MODEL_KIND == "lightgbm" else LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
preds = np.clip(preds, 0, None)
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5
r2 = r2_score(y_test, preds)
within5 = float(np.mean(np.abs(preds - y_test) <= 5) * 100)
within10 = float(np.mean(np.abs(preds - y_test) <= 10) * 100)

print(f"Model: {MODEL_KIND}")
print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")
print(f"MAE:  {mae:.2f} min")
print(f"RMSE: {rmse:.2f} min")
print(f"R^2:  {r2:.3f}")
print(f"Within +-5 min:  {within5:.1f}%")
print(f"Within +-10 min: {within10:.1f}%")

# --- save model + everything the API needs to build features at inference
joblib.dump(model, "model.pkl")
joblib.dump(train_encoder, "train_encoder.pkl")

metrics = {
    "model_kind": MODEL_KIND,
    "features": FEATURES,
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "mae_min": round(mae, 2),
    "rmse_min": round(rmse, 2),
    "r2": round(r2, 3),
    "within_5min_pct": round(within5, 1),
    "within_10min_pct": round(within10, 1),
}
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved model.pkl, train_encoder.pkl, model_metrics.json")
