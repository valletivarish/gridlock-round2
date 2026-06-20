"""
Resource Recommendation Engine — Pillar 2 of the BTP Event Intelligence prototype.

Public API
----------
    from recommend import recommend
    result = recommend(event_dict)   # -> structured dict (see docstring below)

All logic is grounded in the EDA hotspot tables and the impact model outputs.
Thresholds are commented with their data rationale so they can be revisited
as the dataset grows or policies change.
"""

import os
import sys
import math
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_EDA  = os.path.join(_HERE, "eda")
sys.path.insert(0, _HERE)  # import siblings whether files sit at repo root (deploy) or in round_2/ (local)

from impact_models import predict_impact

# ── Load EDA hotspot tables once at import time ───────────────────────────────

_corridors_df   = pd.read_csv(os.path.join(_EDA, "table_top10_corridors.csv"))
_causes_df      = pd.read_csv(os.path.join(_EDA, "table_bottleneck_causes.csv"))
_junctions_df   = pd.read_csv(os.path.join(_EDA, "table_top15_junctions.csv"))

# Corridor hotspot set (top-10 by composite impact score from EDA)
_HOTSPOT_CORRIDORS: set = set(_corridors_df["corridor"].str.strip())

# Corridor -> impact_score lookup (0–1 composite from EDA; fallback = 0)
_CORRIDOR_IMPACT: dict = dict(
    zip(_corridors_df["corridor"].str.strip(), _corridors_df["impact_score"])
)

# Cause -> median clearance minutes (data-driven benchmark, not a prediction)
_CAUSE_MEDIAN_MIN: dict = dict(
    zip(_causes_df["event_cause"].str.strip(), _causes_df["median_min"])
)

# Junction hotspot set (top-15 by event count from EDA)
_HOTSPOT_JUNCTIONS: set = set(_junctions_df["junction"].str.strip())

# ── Corridor -> nearest police station (most-common assignment in real data) ──
# Built from the data; covers all corridors observed in 8 k+ historical events.
_CORRIDOR_STATION: dict = {
    "Airport New South Road": "Hennuru",
    "Bannerghata Road":       "Mico Layout",
    "Bellary Road 1":         "Sadashivanagar",
    "Bellary Road 2":         "Yelahanka",
    "CBD 1":                  "Shivajinagar",
    "CBD 2":                  "Cubbon Park",
    "Hennur Main Road":       "Hennuru",
    "Hosur Road":             "Madiwala",
    "IRR(Thanisandra road)":  "Adugodi",
    "Magadi Road":            "Kamakshipalya",
    "Mysore Road":            "Halasuru Gate",
    "Non-corridor":           "No Police Station",
    "ORR East 1":             "HAL Old Airport",
    "ORR East 2":             "HAL Old Airport",
    "ORR North 1":            "Banaswadi",
    "ORR North 2":            "Jalahalli",
    "ORR West 1":             "Banashankari",
    "Old Airport Road":       "Jeevanbheemanagar",
    "Old Madras Road":        "Banaswadi",
    "Tumkur Road":            "Yeshwanthpura",
    "Varthur Road":           "HAL Old Airport",
    "West of Chord Road":     "Vijayanagara",
}

# Zone -> police station fallback when corridor is unseen
_ZONE_STATION: dict = {
    "Central Zone 1": "Ashok Nagar",
    "Central Zone 2": "Halasuru Gate",
    "East Zone 1":    "HAL Old Airport",
    "East Zone 2":    "K.R. Pura",
    "North Zone 1":   "Hennuru",
    "North Zone 2":   "Yelahanka",
    "South Zone 1":   "Jayanagara",
    "South Zone 2":   "Madiwala",
    "West Zone 1":    "Yeshwanthpura",
    "West Zone 2":    "Sadashivanagar",
}

# Police station lat/lon centroids (median of historical events) for geo-fallback
_STATION_COORDS: dict = {
    "Adugodi":                 (12.931452, 77.621345),
    "Ashok Nagar":             (12.966152, 77.606877),
    "Banashankari":            (12.923867, 77.553411),
    "Banaswadi":               (12.997415, 77.658525),
    "Basavanagudi":            (12.943511, 77.573733),
    "Bellandur":               (12.921682, 77.669015),
    "Byatarayanapura":         (12.952217, 77.537201),
    "Chamarajpet":             (12.963051, 77.567142),
    "Chikkabanavara":          (13.045650, 77.507166),
    "Chikkajala":              (13.165263, 77.636664),
    "City Market":             (12.963384, 77.577632),
    "Cubbon Park":             (12.977745, 77.595424),
    "Devanahalli Airport":     (13.242711, 77.707075),
    "Electronic City":         (12.850914, 77.664640),
    "HAL Old Airport":         (12.956265, 77.700571),
    "HSR Layout":              (12.912197, 77.628728),
    "Halasur":                 (12.971341, 77.625605),
    "Halasuru Gate":           (12.966294, 77.587124),
    "Hebbala":                 (13.042661, 77.593289),
    "Hennuru":                 (13.041164, 77.631782),
    "High ground":             (12.989397, 77.585305),
    "Hulimavu":                (12.876100, 77.597820),
    "J.P. Nagar":              (12.905506, 77.587115),
    "Jalahalli":               (13.043193, 77.547666),
    "Jayanagara":              (12.918617, 77.589460),
    "Jeevanbheemanagar":       (12.978207, 77.641813),
    "Jnanabharathi":           (12.960920, 77.510564),
    "K.G. Halli":              (13.034624, 77.619996),
    "K.R. Pura":               (13.012727, 77.700755),
    "K.S. Layout":             (12.910690, 77.563423),
    "Kamakshipalya":           (12.988274, 77.507915),
    "Kengeri":                 (12.910488, 77.480183),
    "Kodigehalli":             (13.044666, 77.587125),
    "Madiwala":                (12.921288, 77.620424),
    "Magadi Road":             (12.975735, 77.554817),
    "Mahadevapura":            (12.994738, 77.716050),
    "Malleshwaram":            (13.008109, 77.563289),
    "Mico Layout":             (12.914621, 77.601631),
    "No Police Station":       (13.059105, 77.465378),
    "Peenya":                  (13.039985, 77.517034),
    "Pulikeshinagar(F.Town)":  (12.995535, 77.615860),
    "R.T. Nagar":              (13.018948, 77.585537),
    "Rajajinagar":             (13.007508, 77.544070),
    "Sadashivanagar":          (13.010834, 77.583793),
    "Sheshadripuram":          (12.984837, 77.572290),
    "Shivajinagar":            (12.981762, 77.603105),
    "Thalagattapura":          (12.873966, 77.544782),
    "Upparpet":                (12.975882, 77.576242),
    "V.V.Puram (C.Pet)":       (12.958124, 77.573851),
    "Vijayanagara":            (12.975362, 77.542086),
    "Whitefield":              (12.950805, 77.746997),
    "Wilson Garden":           (12.946565, 77.594379),
    "Yelahanka":               (13.099177, 77.597288),
    "Yeshwanthpura":           (13.028353, 77.541764),
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Approximate great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_station_by_coords(lat: float, lon: float) -> str:
    """Return police station name closest to (lat, lon) by Haversine distance."""
    best, best_dist = "Cubbon Park", float("inf")   # CBD centroid as last resort
    for station, (slat, slon) in _STATION_COORDS.items():
        d = _haversine_km(lat, lon, slat, slon)
        if d < best_dist:
            best_dist = d
            best = station
    return best


def _resolve_station(event_dict: dict) -> tuple[str, str]:
    """
    Determine the responding police station and how we found it.
    Priority:  1) corridor lookup  2) zone lookup  3) coordinates  4) hardcoded default
    Returns (station_name, method_label).
    """
    corridor = str(event_dict.get("corridor", "")).strip()
    zone     = str(event_dict.get("zone", "")).strip()
    lat      = event_dict.get("latitude")
    lon      = event_dict.get("longitude")

    if corridor in _CORRIDOR_STATION:
        return _CORRIDOR_STATION[corridor], "corridor lookup"

    if zone in _ZONE_STATION:
        return _ZONE_STATION[zone], "zone lookup"

    if lat and lon:
        try:
            return _nearest_station_by_coords(float(lat), float(lon)), "nearest by coordinates"
        except (TypeError, ValueError):
            pass

    return "Cubbon Park", "default (no location info)"


def _severity_score(
    closure_prob: float,
    priority: str,
    corridor: str,
    junction: str,
    cause: str,
) -> float:
    """
    Compute a 0–1 severity score from four independent signals.

    Weights are deliberately equal (0.25 each) so no single signal dominates
    and the logic is fully auditable.  Each signal is normalised to [0, 1].

    Signal 1 — road_closure_prob [0,1]:  directly from the model (AUC 0.81)
    Signal 2 — priority flag [0 or 1]:   High=1, anything else=0 (62% of events are High)
    Signal 3 — hotspot flag [0 or 1]:    corridor in top-10 EDA list, or junction in top-15
    Signal 4 — cause severity [0, 0.5, 1]: construction/road_conditions long-tail causes = 1;
                                            medium-duration causes = 0.5; fast-clearing = 0
    """
    # Signal 2
    priority_score = 1.0 if str(priority).strip().lower() == "high" else 0.0

    # Signal 3: corridor OR junction is a known hotspot
    hotspot = (
        str(corridor).strip() in _HOTSPOT_CORRIDORS
        or str(junction).strip() in _HOTSPOT_JUNCTIONS
    )
    hotspot_score = 1.0 if hotspot else 0.0

    # Signal 4: cause-based severity
    # Data: construction=296 min, road_conditions=246 min >> others
    # Grouped as: slow (>= 100 min median), medium (40-100 min), fast (< 40 min)
    slow_causes   = {"construction", "road_conditions", "water_logging", "tree_fall"}
    medium_causes = {"others", "congestion", "accident", "vehicle_breakdown"}
    c = str(cause).strip().lower()
    if c in slow_causes:
        cause_score = 1.0
    elif c in medium_causes:
        cause_score = 0.5
    else:
        cause_score = 0.25    # fast-clearing or unknown cause

    return 0.25 * closure_prob + 0.25 * priority_score + 0.25 * hotspot_score + 0.25 * cause_score


def _severity_label(score: float) -> str:
    """
    Map numeric score to Low/Medium/High.
    Thresholds derived from the equal-weight scale:
      < 0.35  -> Low    (e.g. Low priority, non-hotspot, fast cause, low closure prob)
      0.35-0.6 -> Medium
      >= 0.6  -> High   (e.g. High priority hotspot with likely closure)
    """
    if score >= 0.60:
        return "High"
    elif score >= 0.35:
        return "Medium"
    return "Low"


def _officer_count(severity: str, is_peak: int, closure_prob: float) -> int:
    """
    Recommended officer deployment count.

    Base from BTP practice (reasonable approximation; adjust via policy):
      Low    -> 2 officers   (monitor, assist clearance)
      Medium -> 4 officers   (active traffic management)
      High   -> 6 officers   (full response + diversion)

    +2 if peak hour (8-10 AM or 5-8 PM), because simultaneous adjacent incidents are more likely.
    +2 if road closure is likely (>= 0.35) to staff both ends of the closure + diversion point.
    """
    base = {"Low": 2, "Medium": 4, "High": 6}[severity]
    peak_bonus    = 2 if is_peak else 0
    closure_bonus = 2 if closure_prob >= 0.35 else 0
    return base + peak_bonus + closure_bonus


def _diversion_advice(corridor: str, closure_prob: float) -> str:
    """
    Short diversion message for the field officer.
    Only meaningful when closure is likely (>= 0.35).
    """
    if closure_prob < 0.35:
        return "No diversion needed; manage flow in situ."

    known_alternates = {
        "Mysore Road":          "Magadi Road or Chord Road",
        "Bellary Road 1":       "Hennur Main Road or Tumkur Road",
        "Airport New South Road": "Bellary Road 2 or Yelahanka bypass",
        "ORR North 1":          "NH-44 service road or Hennur Road",
        "Varthur Road":         "Whitefield Main Road or ITPL Road",
        "Hennur Main Road":     "Bellary Road 1 or Thanisandra Road",
        "Hosur Road":           "Bannerghata Road or Outer Ring Road East",
        "CBD 1":                "MG Road or Residency Road",
        "Old Airport Road":     "ORR East 1 or HAL Airport Road alternate",
        "CBD 2":                "Cubbon Road or Seshadri Road",
    }
    c = str(corridor).strip()
    alt = known_alternates.get(c, "the nearest parallel arterial")
    return f"Activate diversion via {alt}; post officers at diversion junction."


# ── Public API ────────────────────────────────────────────────────────────────

def recommend(event_dict: dict) -> dict:
    """
    Recommend BTP response resources for a reported traffic event.

    Parameters
    ----------
    event_dict : dict
        Any subset of the fields used by predict_impact (event_cause, event_type,
        corridor, zone, junction, police_station, veh_type, priority, latitude,
        longitude, hour, dow, month, is_peak, is_weekend, is_planned).
        Missing or unseen values are handled gracefully; the function will not crash.

    Returns
    -------
    dict with keys:
        expected_clearance_min  — typical clearance for this cause from EDA data
                                  (a benchmark, not a prediction; honest framing)
        road_closure_prob       — model probability of road closure [0, 1]
        severity                — "Low" / "Medium" / "High"
        recommended_officers    — integer; scaled by severity + peak hour + closure risk
        barricading             — bool; True when closure probability >= 0.35
        diversion_advice        — short string for the field officer
        nearest_station         — responding police station name
        rationale               — one-line plain-English explanation
    """
    # ── 1. Get model predictions ──────────────────────────────────────────────
    impact = predict_impact(event_dict)
    closure_prob = impact["road_closure_prob"]

    # ── 2. Extract event fields (with safe defaults) ──────────────────────────
    cause     = str(event_dict.get("event_cause", "unknown")).strip().lower()
    corridor  = str(event_dict.get("corridor",    "unknown")).strip()
    junction  = str(event_dict.get("junction",    "unknown")).strip()
    priority  = str(event_dict.get("priority",    "unknown")).strip()
    is_peak   = int(event_dict.get("is_peak", 0))

    # ── 3. Expected clearance from EDA cause table (honest benchmark) ─────────
    # Use cause as reported; fallback to overall median (~41 min for vehicle_breakdown,
    # which dominates the dataset at 60% of events).
    expected_clearance = _CAUSE_MEDIAN_MIN.get(cause, 41.0)

    # ── 4. Severity score and label ───────────────────────────────────────────
    score    = _severity_score(closure_prob, priority, corridor, junction, cause)
    severity = _severity_label(score)

    # ── 5. Operational recommendations ───────────────────────────────────────
    officers      = _officer_count(severity, is_peak, closure_prob)
    barricading   = closure_prob >= 0.35   # >35% closure probability warrants pre-staging barricades
    diversion     = _diversion_advice(corridor, closure_prob)
    station, how  = _resolve_station(event_dict)

    # ── 6. Rationale ─────────────────────────────────────────────────────────
    hotspot_flag = corridor in _HOTSPOT_CORRIDORS or junction in _HOTSPOT_JUNCTIONS
    parts = []
    if severity == "High":
        parts.append("High-impact event")
    elif severity == "Medium":
        parts.append("Moderate-impact event")
    else:
        parts.append("Low-impact event")

    if hotspot_flag:
        parts.append(f"{corridor} is a top-10 congestion hotspot")
    if closure_prob >= 0.35:
        parts.append(f"road closure likely ({closure_prob:.0%})")
    parts.append(f"typical clearance for {cause}: ~{int(expected_clearance)} min")
    if is_peak:
        parts.append("occurring during peak hour")

    rationale = "; ".join(parts) + "."

    return {
        "expected_clearance_min": round(expected_clearance, 1),
        "road_closure_prob":      closure_prob,
        "severity":               severity,
        "recommended_officers":   officers,
        "barricading":            barricading,
        "diversion_advice":       diversion,
        "nearest_station":        station,
        "rationale":              rationale,
    }
