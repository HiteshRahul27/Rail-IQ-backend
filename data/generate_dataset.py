"""
Generates data/training_data.csv — the practice dataset described in Step 1.

Two sources, clearly labeled in the `source` column:
  - '12805' (Janmabhoomi Express): REAL historical data, parsed directly from
    the Excel export the user supplied (Aug 1 - Sep 4 2026, 25 stations/day).
  - Everything else: SYNTHETIC data generated with a structured random-walk
    delay model (autocorrelated delay + rush-hour congestion + rare incident
    spikes), built on realistic station/distance/schedule skeletons for 5
    more real Indian Railways trains. This stands in for live GPS + NTES
    history, which aren't reachable from this environment (see Step 1.3 in
    the brief — synthetic is the explicitly sanctioned fallback).

Output columns: train_id, train_name, journey_date, station_no, station_code,
station_name, distance_km, scheduled_time, delay_minutes, actual_time, source
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
EXCEL_PATH = "/mnt/user-data/uploads/Janmabhoomi_12805_Aug01_Sep04_2026__1_.xlsx"

# ---------------------------------------------------------------------------
# 1. Real data: Janmabhoomi Express (12805)
# ---------------------------------------------------------------------------

def load_real_janmabhoomi():
    df = pd.read_excel(EXCEL_PATH)
    rows = []
    for _, r in df.iterrows():
        sched = r["Scheduled Arrival"]
        if pd.isna(sched):
            sched = r["Scheduled Departure"]  # origin station has no arrival time
        delay = r["Arrival Delay (min)"]
        actual = r["Actual Arrival"]
        rows.append({
            "train_id": "12805",
            "train_name": "Janmabhoomi Express",
            "journey_date": r["Journey Date"].strftime("%Y-%m-%d"),
            "station_no": int(r["Station No."]),
            "station_code": r["Code"],
            "station_name": r["Station"],
            "distance_km": int(r["Distance (km)"]),
            "scheduled_time": sched,
            "delay_minutes": None if pd.isna(delay) else float(delay),
            "actual_time": None if pd.isna(actual) else actual,
            "source": "real",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Synthetic data: 5 more trains, realistic station skeletons
# ---------------------------------------------------------------------------

TRAIN_DEFS = {
    "12806": {
        "name": "Andhra Express", "dep": "09:00", "speed_kmph": 55,
        "stops": [("VSKP", "Visakhapatnam Jn", 0), ("BZA", "Vijayawada Jn", 120),
                  ("WL", "Warangal", 300), ("KZJ", "Kazipet Jn", 320),
                  ("NGP", "Nagpur", 640), ("BPL", "Bhopal Jn", 980),
                  ("NDLS", "New Delhi", 1670)],
    },
    "12727": {
        "name": "Godavari Express", "dep": "06:15", "speed_kmph": 58,
        "stops": [("VSKP", "Visakhapatnam Jn", 0), ("RJY", "Rajahmundry", 80),
                  ("KMT", "Khammam", 260), ("WL", "Warangal", 300),
                  ("SC", "Secunderabad Jn", 500)],
    },
    "12841": {
        "name": "Coromandel Express", "dep": "18:50", "speed_kmph": 62,
        "stops": [("MAS", "Chennai Central", 0), ("VSKP", "Visakhapatnam Jn", 780),
                  ("BBS", "Bhubaneswar", 1100), ("KGP", "Kharagpur Jn", 1450),
                  ("HWH", "Howrah Jn", 1660)],
    },
    "20704": {
        "name": "Vande Bharat Express", "dep": "05:45", "speed_kmph": 90,
        "stops": [("SC", "Secunderabad Jn", 0), ("KMT", "Khammam", 180),
                  ("RJY", "Rajahmundry", 350), ("VSKP", "Visakhapatnam Jn", 500)],
    },
    "22691": {
        "name": "Rajdhani Express", "dep": "20:30", "speed_kmph": 65,
        "stops": [("BNC", "Bengaluru Cant", 0), ("GTL", "Guntakal Jn", 300),
                  ("BPL", "Bhopal Jn", 1780), ("NDLS", "New Delhi", 2365)],
    },
}

DATE_RANGE = pd.date_range("2026-08-01", "2026-09-04", freq="D")


def simulate_delay_walk(n_stops, rush_hours):
    """Autocorrelated delay random walk with rush-hour congestion and a small
    chance of a major incident (mirrors the real Janmabhoomi Aug-31 outlier)."""
    delays = [max(0, round(RNG.normal(2, 2)))]
    incident_day = RNG.random() < 0.06  # ~6% of journeys hit a major incident
    incident_at = RNG.integers(1, n_stops) if incident_day else -1
    for i in range(1, n_stops):
        drift = RNG.normal(1.5, 4)
        congestion = RNG.normal(4, 2) if rush_hours[i] else RNG.normal(0, 1.5)
        incident = RNG.uniform(40, 160) if i == incident_at else 0
        nxt = max(0, delays[-1] * 0.82 + drift + congestion + incident)
        delays.append(round(nxt))
    return delays


def generate_synthetic():
    all_rows = []
    for train_id, spec in TRAIN_DEFS.items():
        dep_h, dep_m = map(int, spec["dep"].split(":"))
        stops = spec["stops"]
        n = len(stops)
        offsets_min = [d / spec["speed_kmph"] * 60 + i * 3 for i, (_, _, d) in enumerate(stops)]

        for date in DATE_RANGE:
            base_dep = datetime(date.year, date.month, date.day, dep_h, dep_m)
            sched_times = [base_dep + timedelta(minutes=o) for o in offsets_min]
            rush = [t.hour in (7, 8, 9, 17, 18, 19, 20) for t in sched_times]
            delays = simulate_delay_walk(n, rush)

            for i, (code, name, dist) in enumerate(stops):
                sched = sched_times[i]
                delay = delays[i]
                actual = sched + timedelta(minutes=delay)
                all_rows.append({
                    "train_id": train_id,
                    "train_name": spec["name"],
                    "journey_date": date.strftime("%Y-%m-%d"),
                    "station_no": i + 1,
                    "station_code": code,
                    "station_name": name,
                    "distance_km": dist,
                    "scheduled_time": sched,
                    "delay_minutes": float(delay),
                    "actual_time": actual,
                    "source": "synthetic",
                })
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    real_df = load_real_janmabhoomi()
    synth_df = generate_synthetic()
    full = pd.concat([real_df, synth_df], ignore_index=True)
    full.to_csv("data/training_data.csv", index=False)

    print(f"Real rows (12805):      {len(real_df)}")
    print(f"Synthetic rows (5 trains): {len(synth_df)}")
    print(f"Total:                   {len(full)}")
    print(f"Trains: {sorted(full['train_id'].unique())}")
    print(f"Date range: {full['journey_date'].min()} to {full['journey_date'].max()}")
