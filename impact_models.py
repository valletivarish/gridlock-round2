"""
Impact models inference module.

Usage:
    from impact_models import predict_impact
    result = predict_impact({
        "event_cause": "Accident",
        "event_type": "Unplanned",
        "corridor": "ORR",
        "zone": "East",
        "junction": "SilkBoardJunc",
        "police_station": "Madiwala",
        "veh_type": "Car",
        "priority": "High",
        "latitude": 12.917,
        "longitude": 77.623,
        "hour": 9,
        "dow": 1,
        "month": 3,
        "is_peak": 1,
        "is_weekend": 0,
        "is_planned": 0,
    })
    # -> {"duration_min": 68.4, "road_closure_prob": 0.31}
"""

import os
import numpy as np
import pandas as pd
import joblib

_DIR = os.path.dirname(__file__)
_MODELS_DIR = os.path.join(_DIR, "models")

# ── lazy-loaded artifact cache ────────────────────────────────────────────────
_cache: dict = {}

def _load():
    if _cache:
        return
    _cache["meta"]         = joblib.load(os.path.join(_MODELS_DIR, "meta.pkl"))
    _cache["dur_encoders"] = joblib.load(os.path.join(_MODELS_DIR, "dur_encoders.pkl"))
    _cache["clf_encoders"] = joblib.load(os.path.join(_MODELS_DIR, "clf_encoders.pkl"))

    best_dur = _cache["meta"]["best_dur_model"]
    best_clf = _cache["meta"]["best_clf_model"]

    _cache["dur_model"] = joblib.load(
        os.path.join(_MODELS_DIR, f"{'lgb' if best_dur == 'lgb' else 'cb'}_duration.pkl")
    )
    _cache["clf_model"] = joblib.load(
        os.path.join(_MODELS_DIR, f"{'lgb' if best_clf == 'lgb' else 'cb'}_closure.pkl")
    )
    _cache["is_catboost_dur"] = best_dur == "catboost"
    _cache["is_catboost_clf"] = best_clf == "catboost"


# ── helpers ───────────────────────────────────────────────────────────────────

_FEATURES = [
    "event_cause", "event_type", "is_planned",
    "corridor", "zone", "junction", "police_station",
    "veh_type", "priority",
    "latitude", "longitude",
    "hour", "dow", "month", "is_peak", "is_weekend",
]
_CAT_COLS = ["event_cause", "event_type", "corridor", "zone",
             "junction", "police_station", "veh_type", "priority"]
_NUM_COLS = [f for f in _FEATURES if f not in _CAT_COLS]


def _coerce_row(event_dict: dict) -> pd.DataFrame:
    """Coerce an event dict into a single-row DataFrame with correct types."""
    row = {k: event_dict.get(k, None) for k in _FEATURES}
    # fill missing categoricals with 'unknown' so encoders can map them
    for c in _CAT_COLS:
        if row[c] is None or (isinstance(row[c], float) and np.isnan(row[c])):
            row[c] = "unknown"
        else:
            row[c] = str(row[c])
    # fill missing numerics with 0
    for c in _NUM_COLS:
        if row[c] is None:
            row[c] = 0
    return pd.DataFrame([row])


def _apply_lgb_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply saved OOF target encoders; unseen categories get global mean."""
    parts = {}
    for col in _CAT_COLS:
        enc = encoders[col]
        parts[col + "_enc"] = df[col].map(enc["mapping"]).fillna(enc["global_mean"]).values
    enc_df = pd.DataFrame(parts)
    return pd.concat([enc_df, df[_NUM_COLS].reset_index(drop=True)], axis=1)


# ── public API ────────────────────────────────────────────────────────────────

def predict_impact(event_dict: dict) -> dict:
    """
    Predict event impact from fields known at report time.

    Parameters
    ----------
    event_dict : dict
        Keys matching any subset of _FEATURES.  Unknown/missing keys
        default to 'unknown' (categoricals) or 0 (numerics).
        Unseen category values are handled gracefully via global-mean fallback.

    Returns
    -------
    dict with:
        duration_min       — predicted event duration in minutes
        road_closure_prob  — probability of road closure [0, 1]
    """
    _load()

    df = _coerce_row(event_dict)

    # ── duration ──────────────────────────────────────────────────────────────
    if _cache["is_catboost_dur"]:
        # CatBoost handles unseen categories natively
        log_pred = _cache["dur_model"].predict(df[_FEATURES])
    else:
        X = _apply_lgb_encoders(df, _cache["dur_encoders"])
        log_pred = _cache["dur_model"].predict(X)

    duration_min = float(np.expm1(log_pred[0]))
    duration_min = max(0.0, duration_min)  # ensure non-negative

    # ── road closure ─────────────────────────────────────────────────────────
    if _cache["is_catboost_clf"]:
        prob = _cache["clf_model"].predict_proba(df[_FEATURES])[:, 1]
    else:
        X = _apply_lgb_encoders(df, _cache["clf_encoders"])
        prob = _cache["clf_model"].predict_proba(X)[:, 1]

    road_closure_prob = float(np.clip(prob[0], 0.0, 1.0))

    return {
        "duration_min":      round(duration_min, 1),
        "road_closure_prob": round(road_closure_prob, 4),
    }
