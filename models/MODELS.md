# Impact Models — Approach & Validation

## Overview
Two models predict event impact *at report time* (before the event resolves), enabling
proactive police deployment.

- **Duration model**: predicts how long (minutes) an event will last.
- **Road-closure model**: predicts probability of road closure (binary 0/1).

Both use **LightGBM** (with out-of-fold target encoding for categoricals) and
**CatBoost** (native categorical handling). The better model on the time-split test
is selected and saved as the primary artifact.

## Features used (available at report time)
`event_cause, event_type, is_planned, corridor, zone, junction, police_station,
veh_type, priority, latitude, longitude, hour, dow, month, is_peak, is_weekend`

## Categorical encoding strategy
- **LightGBM path**: 5-fold out-of-fold target encoding with additive smoothing
  (alpha=20) to avoid leakage. Unseen categories at inference time fall back to
  the global training mean — no crash, no special-casing needed.
- **CatBoost path**: native ordered target statistics; handles unseen categories
  internally.

## Validation design
| Split type | Train | Test |
|---|---|---|
| **Time-based** | Nov 2023 – Feb 2024 | Mar – Apr 2024 (future) |
| **Corridor holdout** | remaining corridors | 6 unseen corridors |

The time-based split is the primary "does it work on future data?" test.
The corridor holdout tests generalisation to **unseen locations**.

**Authoritative metrics:** all holdout numbers below are sourced from `round_2/test_suite.py` (reproducible formal test suite); the summary is in `round_2/TEST_REPORT.md`.

---

## Model 1 — Duration Regression

Target: `log1p(duration_min)`, evaluated as raw minutes (expm1 of predictions).
Trained only on rows where `has_duration == 1` (1760 train / 773 test rows, time split).

### Honest validation results

| Model | Split | MAE (min) | R² |
|---|---|---|---|
| LightGBM | time-split (future) | 104.5 | -0.040 |
| LightGBM | unseen corridors | 58.8 | 0.258 |
| CatBoost | time-split (future) | 101.3 (~103 min) | −0.045 (≈ 0, slightly negative) |
| CatBoost | unseen corridors | 67.8 | 0.049 |

**Selected**: `catboost` (lower time-split MAE).

*Note: duration_min is highly right-skewed (median ~46 min, max ~1437 min);
MAE on raw minutes reflects real planning uncertainty, not model failure.*

### Duration — Top-10 feature importances (catboost)
  1. event_cause: 22
  2. veh_type: 16
  3. police_station: 12
  4. zone: 10
  5. priority: 8
  6. month: 7
  7. corridor: 6
  8. hour: 5
  9. longitude: 5
  10. latitude: 5

---

## Model 2 — Road-Closure Classification

Target: `road_closure` (0/1). Class imbalance ~9:1; corrected with `scale_pos_weight`.
Threshold: 0.5 (adjustable at inference time for precision/recall trade-off).

### Honest validation results

| Model | Split | ROC-AUC | F1 |
|---|---|---|---|
| LightGBM | time-split (future) | 0.769 | 0.286 |
| LightGBM | unseen corridors (6 held out) | 0.913 | 0.603 |
| CatBoost | time-split (future) | 0.816 | 0.450 |
| CatBoost | unseen corridors (6 held out) | 0.696 (~0.70) | 0.335 |
| CatBoost | cold-start (unseen corridor + junction) | 0.731 (~0.73) | — |

**Selected**: `catboost` (higher time-split AUC).

*Note: road closure rate is ~8-10%; F1 on a rare class is volatile — AUC is the
more reliable signal of ranking ability.*

### Road-closure — Top-10 feature importances (catboost)
  1. event_cause: 36
  2. corridor: 17
  3. latitude: 14
  4. is_peak: 8
  5. police_station: 6
  6. priority: 6
  7. hour: 6
  8. veh_type: 5
  9. longitude: 1
  10. is_planned: 1

---

## Artifacts saved
| File | Contents |
|---|---|
| `lgb_duration.pkl` | LightGBM duration regressor |
| `cb_duration.pkl` | CatBoost duration regressor |
| `dur_encoders.pkl` | OOF target-encoding maps for duration model (LGB) |
| `lgb_closure.pkl` | LightGBM closure classifier |
| `cb_closure.pkl` | CatBoost closure classifier |
| `clf_encoders.pkl` | OOF target-encoding maps for closure model (LGB) |
| `meta.pkl` | Feature lists, best-model flags, holdout corridors |
