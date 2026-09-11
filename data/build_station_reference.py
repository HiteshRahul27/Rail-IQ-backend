"""Builds data/stations_reference.json — one canonical station list per train
(code, name, distance_km, scheduled offset from origin) used by the API to
compute inference features from just a train_id + station_no."""
import pandas as pd
import json

df = pd.read_csv("data/training_data.csv", parse_dates=["scheduled_time"], dtype={"train_id": str})
df = df.sort_values(["train_id", "journey_date", "station_no"])

reference = {}
for train_id, grp in df.groupby("train_id"):
    first_date = grp["journey_date"].min()
    day = grp[grp["journey_date"] == first_date].sort_values("station_no")
    origin_time = day["scheduled_time"].iloc[0]
    stations = []
    for _, r in day.iterrows():
        offset_min = (r["scheduled_time"] - origin_time).total_seconds() / 60
        stations.append({
            "station_no": int(r["station_no"]),
            "code": r["station_code"],
            "name": r["station_name"],
            "distance_km": int(r["distance_km"]),
            "scheduled_offset_min": round(offset_min, 1),
            "scheduled_hour": int(r["scheduled_time"].hour),
        })
    reference[train_id] = {
        "train_name": grp["train_name"].iloc[0],
        "stations": stations,
    }

with open("data/stations_reference.json", "w") as f:
    json.dump(reference, f, indent=2)

print("Trains:", list(reference.keys()))
for tid, v in reference.items():
    print(f"  {tid} {v['train_name']}: {len(v['stations'])} stations")
