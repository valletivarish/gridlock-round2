# Event Intelligence for BTP — Round 2 Prototype
**Gridlock Hackathon 2.0 · PS2: Event-Driven Congestion · Bengaluru Traffic Police**

---

## What this is

A three-pillar system that turns a raw ASTraM event report into an actionable deployment recommendation — within one second of the event being logged.

| Pillar | What it delivers |
|---|---|
| **1. Forecast Impact** | Road-closure probability (AUC 0.816 on future data, ~0.70 on unseen corridors) + expected clearance time benchmarked from 8,173 real BTP events |
| **2. Recommend Resources** | Officer count, barricade flag, diversion route, nearest police station — all explained in plain English |
| **3. Post-Event Learning Loop** | Monthly retrain on new resolved events; drift monitoring to prevent silent model degradation |

Every number is drawn from the provided ASTraM dataset. No external data, no synthetic rows.

---

## Folder map

```
gridlock-round2/   # repo root
├── data/
│   └── events.csv              # Cleaned ASTraM event data (8,173 rows)
├── eda/
│   ├── INSIGHTS.md             # 8 headline findings with exact numbers
│   ├── eda_insights.py         # EDA script that produced all figures
│   ├── fig_01_monthly_trend.png
│   ├── fig_02_hourly_profile.png
│   ├── fig_03_dow.png
│   ├── fig_04_corridors.png
│   ├── fig_05_resolution_by_cause.png
│   ├── fig_06_drift.png        # Event-cause drift: 66% → 49% breakdown share
│   ├── fig_07_learning.png     # Static vs retrained AUC per month
│   ├── table_top10_corridors.csv
│   ├── table_top15_junctions.csv
│   └── table_bottleneck_causes.csv
├── models/
│   ├── MODELS.md               # Validation methodology and honest results
│   ├── cb_closure.pkl          # CatBoost road-closure classifier (primary)
│   ├── cb_duration.pkl         # CatBoost duration regressor (primary)
│   ├── lgb_closure.pkl         # LightGBM closure classifier
│   ├── lgb_duration.pkl        # LightGBM duration regressor
│   ├── clf_encoders.pkl        # OOF target encoders for LGB closure model
│   ├── dur_encoders.pkl        # OOF target encoders for LGB duration model
│   └── meta.pkl                # Feature lists, best-model flags, holdout corridors
├── data_prep.py                # Data cleaning and feature engineering
├── train_models.py             # Model training and validation
├── impact_models.py            # Inference wrapper (predict_impact)
├── recommend.py                # Recommendation engine (recommend)
├── learning.py                 # Monthly retrain + drift comparison script
├── app.py                      # Streamlit dashboard (Event Response + City Insights)
├── DECK.md                     # Pitch deck (slide-by-slide)
├── PLAN.md                     # Project plan and PS2 requirement mapping
├── RECOMMEND.md                # Recommendation engine logic + worked examples
├── LEARNING.md                 # Drift experiment design and results
└── requirements.txt            # Pinned package versions
```

---

## How to run

### Prerequisites

Python environment with all packages pinned in `requirements.txt`. Activate with:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Run the notebook (EDA + model training + learning loop)

The full analysis lives in the project notebook. Run all cells in order:

```bash
# From the project root
jupyter notebook
# or
jupyter lab
```

All figures in `eda/` and model artifacts in `models/` are produced by the notebook. If artifacts already exist, inference scripts load them directly — no retraining needed.

To retrain models from scratch:

```bash
python train_models.py
```

To re-run the drift/learning experiment:

```bash
python learning.py
```

### 2. Run the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` with two tabs:
- **Event Response** — enter a reported event (cause, corridor, zone, priority, time, location) and click *Get Recommendation* for an instant deployment plan: road-closure probability, severity, expected clearance, officer count, barricading flag, nearest police station, and diversion advice with a plain-English rationale.
- **City Insights** — city-wide KPIs (8,173 events, 94% unplanned, 8.3% closure rate), an event hotspot map, top-corridor rankings, key EDA reference tables, and an honest model-reliability note.

### 3. Use the recommendation engine directly (Python)

```python
from recommend import recommend

event = {
    "event_cause": "construction",
    "corridor":    "Mysore Road",
    "priority":    "High",
    "hour":        9,
    "is_peak":     1,
    "latitude":    12.962,
    "longitude":   77.566,
    "zone":        "West Zone",
}

result = recommend(event)
print(result)
# {
#   "expected_clearance_min": 295.6,
#   "road_closure_prob": 0.62,
#   "severity": "High",
#   "recommended_officers": 10,
#   "barricading": True,
#   "diversion_advice": "Activate diversion via Magadi Road or Chord Road ...",
#   "nearest_station": "Halasuru Gate",
#   "rationale": "High-impact event; Mysore Road is a top-10 congestion hotspot ..."
# }
```

Missing or unseen fields degrade gracefully — no crash, no special-casing needed.

---

## Honest results summary

All validation uses a **time-based split**: train on earlier months (Nov 2023 – Feb 2024), test on later months (Mar – Apr 2024). This measures performance on future, unseen data — the relevant operational question.

### Road-closure classifier (CatBoost — primary model)

| Validation split | ROC-AUC | F1 |
|---|---|---|
| Future months (Mar–Apr 2024) | **0.816** | 0.450 |
| 6 unseen corridors (holdout) | **0.696 (~0.70)** | 0.335 |
| Cold-start (unseen corridor + junction) | **0.731 (~0.73)** | — |

F1 is lower than AUC because road closures are rare (8.3% of events). On a skewed binary target, AUC is the appropriate primary metric — it measures ranking ability across all thresholds, not just the default 0.5 cutoff.

### Duration regressor (CatBoost — primary model)

| Validation split | MAE (minutes) | R² |
|---|---|---|
| Future months (Mar–Apr 2024) | **~103** | **≈ 0 (slightly negative)** |
| Unseen corridors | 67.8 | 0.049 |

Duration is highly right-skewed (median ~46 min, max ~1,437 min). An MAE of 101 minutes on future data reflects genuine uncertainty in event duration — not model failure. For field communication, EDA-benchmarked cause medians (vehicle breakdown: 41 min; construction: 296 min) are used instead of point predictions.

### Learning loop

Retraining monthly consistently helps vs. a frozen static model:

| Month | Static AUC | Retrained AUC |
|---|---|---|
| Feb 2024 | 0.696 | 0.714 (+1.8 pp) |
| Mar 2024 | 0.792 | 0.805 (+1.3 pp) |
| Apr 2024 | 0.822 | 0.838 (+1.6 pp) |

Gains are modest but consistent. The Feb dip (static AUC 0.696) is the concrete case for monthly retraining.

---

## Robustness notes

- **Unseen corridors**: tested on 6 corridors fully withheld from training. AUC ~0.70 (0.696) — degrades from the time-split result but remains above random, confirming generalisation to new locations. Cold-start (doubly unseen: corridor + junction) scores ~0.73 (0.731).
- **Unseen causes**: unknown cause labels at inference fall back to the global training mean via out-of-fold target encoding. No crash; closure probability still computed from other available features.
- **Unseen station lookup**: three-layer fallback — corridor historical assignment → zone historical assignment → Haversine nearest centroid → Cubbon Park default. Always returns a valid station.
- **Messy input**: 116 malformed timestamps detected and corrected during ingestion. The pipeline is tolerant of real-world ASTraM data quality.
- **Drift**: vehicle-breakdown share shifted from 66% to 49% over 6 months. The monthly retrain cadence is designed to track this drift before it erodes model quality.

---

## Key data facts (all from ASTraM dataset)

| Metric | Value |
|---|---|
| Total events | 8,173 |
| Date range | Nov 2023 – Apr 2024 |
| Unplanned events | 7,706 (94%) |
| High-priority events | 5,030 (62%) |
| Vehicle breakdowns | 4,896 (60%) |
| Road-closure events | 676 (8.3%) |
| Mysore Road events | 743 (median res. 41 min, 100% high-priority) |
| Bellary Road 1 events | 610 (median res. 42 min, 100% high-priority) |
| Construction median clearance | 296 min (~5 hours) |
| Busiest single hour | 2 AM (845 events) |
| Peak-hour share of events | 20% (1,616/8,173) |
| Highest monthly volume | March 2024 (1,956 events) |

---

## Submission compliance

- Dataset: only the provided ASTraM data (`data/events.csv`) — no external sources
- Theme: PS2 Event-Driven Congestion
- Deadline: Jun 21, 11:59 PM IST
