"""
Load + clean the ASTraM event dataset for the Event Intelligence prototype.
Shared by the analysis notebook and the dashboard so cleaning never drifts.
"""
import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "data", "events.csv")

MAX_DURATION_MIN = 24 * 60          # events "open" longer than a day = stale records, not real duration
PEAK_HOURS = {8, 9, 10, 17, 18, 19, 20}
CAT_COLS = ["event_type", "event_cause", "corridor", "zone", "junction",
            "police_station", "veh_type", "priority"]


def load_events(path=DATA_PATH):
    return pd.read_csv(path)


def clean(df):
    """Parse times, derive event duration + time features, standardise categoricals."""
    df = df.copy()

    # --- timestamps (UTC -> IST for local hour-of-day) ---
    # ISO8601 parsing handles varying precision (with/without microseconds) robustly,
    # which matters for real/streaming data where formats are inconsistent.
    for c in ["start_datetime", "resolved_datetime", "closed_datetime", "created_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True, format="ISO8601")
    df["has_start_time"] = df["start_datetime"].notna().astype(int)

    # --- event duration: time from start to resolved (or closed), in minutes ---
    end = df["resolved_datetime"].fillna(df["closed_datetime"])
    dur = (end - df["start_datetime"]).dt.total_seconds() / 60.0
    dur[(dur < 0) | (dur > MAX_DURATION_MIN)] = np.nan      # drop negatives + stale-open records
    df["duration_min"] = dur
    df["has_duration"] = df["duration_min"].notna().astype(int)

    # --- time-of-day features (IST) ---
    s = df["start_datetime"].dt.tz_convert("Asia/Kolkata")
    df["hour"] = s.dt.hour
    df["dow"] = s.dt.dayofweek
    df["month"] = s.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_peak"] = df["hour"].isin(PEAK_HOURS).astype(int)

    # --- target / flag columns ---
    df["is_planned"] = (df["event_type"] == "planned").astype(int)
    df["road_closure"] = df["requires_road_closure"].fillna(False).astype(bool).astype(int)
    df["is_high_priority"] = (df["priority"] == "High").astype(int)

    # --- standardise categoricals ---
    for c in CAT_COLS:
        df[c] = df[c].astype("object").fillna("unknown").astype(str).str.strip()

    return df


def load_clean(path=DATA_PATH):
    return clean(load_events(path))
