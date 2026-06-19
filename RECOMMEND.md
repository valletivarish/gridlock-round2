# Resource Recommendation Engine — Logic & Examples

**Module:** `round_2/recommend.py`  
**Entry point:** `recommend(event_dict) -> dict`

---

## What it does

Given a reported event (any subset of the fields used by the impact models), it returns:

| Output field | Meaning |
|---|---|
| `expected_clearance_min` | Typical clearance time for this cause, from EDA data — a benchmark, not a prediction |
| `road_closure_prob` | Model probability of road closure (AUC 0.81 classifier) |
| `severity` | Low / Medium / High — composite score across four signals |
| `recommended_officers` | Integer officer count, scaled by severity + peak-hour + closure risk |
| `barricading` | `True` when closure probability ≥ 0.35 — pre-stage barricades |
| `diversion_advice` | Short field instruction; names alternative route if closure is likely |
| `nearest_station` | Responding police station (corridor lookup → zone → geo-proximity) |
| `rationale` | One-line plain-English explanation for the field officer |

---

## Logic walkthrough

### Step 1 — Impact model predictions

Calls `predict_impact(event_dict)` from `impact_models.py`:
- `road_closure_prob` — well-calibrated (AUC 0.81); used directly in severity scoring.
- `duration_min` — a typical-for-cause benchmark; **not exposed** in the output because the exact duration of any single event is not reliably predictable. The EDA cause table provides cleaner median benchmarks for field communication.

### Step 2 — Expected clearance (honest benchmark)

Looked up from `eda/table_bottleneck_causes.csv` by `event_cause`.  
Key values from data (8,173 real BTP events):

| Cause | Median clearance | P75 clearance |
|---|---|---|
| construction | 296 min | 427 min |
| road_conditions | 246 min | 756 min |
| water_logging | 107 min | 283 min |
| tree_fall | 90 min | 331 min |
| accident | 41 min | 62 min |
| vehicle_breakdown | 41 min | 74 min |

Unseen causes fall back to **41 min** (the vehicle_breakdown median, which represents 60% of all events).

### Step 3 — Severity score

Four signals, **equal weight (0.25 each)**, so no single signal dominates:

| Signal | Source | Value |
|---|---|---|
| `road_closure_prob` | Impact model | 0–1 continuous |
| `priority` flag | Event field | High=1, else=0 |
| Hotspot flag | EDA top-10 corridors + top-15 junctions | 1 if in hotspot set, else 0 |
| Cause severity | EDA clearance table | slow causes (≥100 min)=1.0, medium=0.5, fast/unknown=0.25 |

Severity label thresholds:
- **High**: score ≥ 0.60
- **Medium**: 0.35 – 0.59
- **Low**: < 0.35

### Step 4 — Officer count

Base count reflects BTP minimum deployment needed to manage each severity tier:

| Severity | Base officers |
|---|---|
| Low | 2 (monitor + assist clearance) |
| Medium | 4 (active traffic management) |
| High | 6 (full response + diversion) |

Bonuses (additive, each justified by operational load):
- **+2** during peak hours (8–10 AM, 5–8 PM) — simultaneous nearby incidents are more likely
- **+2** when closure probability ≥ 0.35 — staff both ends of closure plus diversion point

### Step 5 — Barricading threshold

`barricading = True` when `road_closure_prob ≥ 0.35`.

Rationale: 8.3% of all events result in road closure (676/8,173). A 35% model probability represents a 4× uplift over the base rate — high enough to pre-stage barricades (low cost if wrong) before the closure is confirmed.

### Step 6 — Nearest station resolution

Priority chain to find the responding station:
1. **Corridor lookup** — most-common station historically assigned to that corridor (from real data)
2. **Zone lookup** — most-common station for the zone
3. **Nearest by coordinates** — Haversine distance to station lat/lon centroids from data
4. **Default** — Cubbon Park (central CBD fallback)

All three layers ensure the function never crashes on unseen corridors.

---

## Robustness

- Missing or unseen event fields default gracefully (unknown categoricals → global mean; missing numerics → 0)
- Unseen corridors fall through to zone → geo-proximity → default chain
- Unseen causes fall back to the dataset-wide vehicle_breakdown median
- No hard-coded crashes on any input combination

---

## Worked Examples

### Example 1 — Minor vehicle breakdown (Bannerghata Road, Low priority, off-peak)

```python
event = {
    "event_cause": "vehicle_breakdown",
    "corridor":    "Bannerghata Road",
    "zone":        "South Zone 1",
    "priority":    "Low",
    "latitude":    12.899, "longitude": 77.599,
    "hour": 14, "is_peak": 0, ...
}
recommend(event)
```

```
expected_clearance_min   41.1
road_closure_prob        0.44          # model sees Bannerghata + car + afternoon
severity                 Low           # Low priority + no hotspot + fast cause
recommended_officers     4             # base 2 + 2 (closure likely per model)
barricading              True          # 44% > 0.35 threshold
diversion_advice         Activate diversion via use nearest parallel arterial; ...
nearest_station          Mico Layout   # most common station for Bannerghata Road
rationale                Low-impact event; road closure likely (44%); typical
                         clearance for vehicle_breakdown: ~41 min.
```

**Note:** The severity is Low despite the 44% closure probability because the Low priority + non-hotspot + fast-clearing cause all pull the composite score down. The 44% closure probability is from the model, not from human judgment — the barricade flag and +2 officers ensure the closure scenario is covered even though overall severity is Low.

---

### Example 2 — Construction on Mysore Road (likely closure, peak hour)

```python
event = {
    "event_cause": "construction",
    "corridor":    "Mysore Road",
    "junction":    "SilkBoardJunc",
    "priority":    "High",
    "hour": 9, "is_peak": 1,
    "latitude": 12.962, "longitude": 77.566, ...
}
recommend(event)
```

```
expected_clearance_min   295.6         # EDA: construction median = 296 min (~5 hours)
road_closure_prob        0.62          # model output
severity                 High          # all four signals fire: high prob + High priority
                                       # + top-10 hotspot + slow cause
recommended_officers     10            # base 6 + 2 (peak) + 2 (closure)
barricading              True
diversion_advice         Activate diversion via Magadi Road or Chord Road; ...
nearest_station          Halasuru Gate # standard station for Mysore Road
rationale                High-impact event; Mysore Road is a top-10 congestion hotspot;
                         road closure likely (62%); typical clearance for construction:
                         ~295 min; occurring during peak hour.
```

This is the worst-case scenario the engine is designed to catch: a long-duration event on the city's highest-burden corridor during the morning rush.

---

### Example 3 — Unseen corridor + unseen cause (Fog, NH-44 Bypass)

```python
event = {
    "event_cause": "Fog / Low Visibility",   # rare cause, only 2 events in data
    "corridor":    "NH-44 Bypass",            # not in any historical table
    "zone":        "North Zone 1",
    "priority":    "High",
    "latitude":    13.10, "longitude": 77.59,
    "hour": 3, "is_peak": 0, ...
}
recommend(event)
```

```
expected_clearance_min   41.0          # fallback to vehicle_breakdown median (base rate)
road_closure_prob        0.55          # model uses zone + coordinates + priority
severity                 Medium        # High priority fires but no hotspot match,
                                       # unknown cause scores 0.25 not 1.0
recommended_officers     6             # base 4 (Medium) + 2 (closure likely)
barricading              True          # 55% > 0.35
diversion_advice         Activate diversion via use nearest parallel arterial; ...
nearest_station          Hennuru       # zone lookup: North Zone 1 → Hennuru
rationale                Moderate-impact event; road closure likely (55%); typical
                         clearance for fog / low visibility: ~41 min.
```

No crash. The unseen corridor degrades gracefully to zone-based station lookup. The clearance benchmark falls back to 41 min (honest: we have no clearance data for this cause). The model still returns a meaningful closure probability using the available context (coordinates, zone, priority).

---

## Limitations

- **Duration benchmark, not a prediction.** The `expected_clearance_min` is the historical median for this cause class. Individual events vary widely (vehicle_breakdown P75 = 74 min, construction P75 = 427 min).
- **Officer counts are starting points.** The numbers represent minimum effective deployment; supervisors should adjust based on ground conditions and simultaneous incident load.
- **Diversion routes need field verification.** The alternate routes listed are based on road network knowledge; construction diversions or flooded alternates may reduce options.
- **Model drift.** The event-cause mix is shifting (breakdown share fell from 66% to 49% over 6 months). Thresholds should be reviewed monthly.
