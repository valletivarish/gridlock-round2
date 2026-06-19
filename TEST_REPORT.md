# BTP Event Intelligence — Test Report
**Flipkart Gridlock Round-2 · PS2 (Bengaluru Traffic Police)**
Run: `python round_2/test_suite.py`
Result: **78 / 78 tests passed · 0 failed**

---

## Section Breakdown

| Section | Tests | Result |
|---------|------:|--------|
| (1) Data / Cleaning edge cases | 32 | ✓ All pass |
| (2) Accuracy on unseen data | 7 | ✓ All pass |
| (3) Model robustness — `predict_impact` | 12 | ✓ All pass |
| (4) Recommendation robustness — `recommend` | 24 | ✓ All pass |
| (5) Dashboard smoke test — `AppTest` | 3 | ✓ All pass |

---

## (1) Data / Cleaning Edge Cases

| # | Edge case | Status |
|---|-----------|--------|
| 1.1 | Row count ≥ 8,000 (got 8,173) | ✓ |
| 1.2 | Duration recompute matches manual (max delta = 0.0) | ✓ |
| 1.3 | Timestamps **with** microseconds parse without NaT | ✓ |
| 1.4 | Timestamps **without** microseconds parse without NaT | ✓ |
| 1.5 | Timestamps with `+00` (no colon) suffix parse without NaT | ✓ |
| 1.6 | `hour` in [0, 23] — no NaN | ✓ |
| 1.7 | `dow` in [0, 6] — no NaN | ✓ |
| 1.8 | `month` in [1, 12] — no NaN | ✓ |
| 1.9 | `is_planned` matches raw `event_type=='planned'` count (467) | ✓ |
| 1.10 | `road_closure` matches raw `requires_road_closure` count (676) | ✓ |
| 1.11–1.18 | All 8 categorical columns never NaN on real data | ✓ |
| 1.19 | `clean()` does not crash on hand-built edge-case DataFrame | ✓ |
| 1.20 | Missing `corridor` → fills to `'unknown'` | ✓ |
| 1.21 | Missing `zone` → fills to `'unknown'` | ✓ |
| 1.22 | **Negative duration** (resolved before start) → `duration_min = NaN` | ✓ |
| 1.23 | **>24 h duration** (stale open record) → `duration_min = NaN` | ✓ |
| 1.24 | Unseen `event_cause` value (`'alien_invasion'`) survives `clean()` as-is | ✓ |
| 1.25–1.32 | All 8 categorical columns never NaN on edge-case DataFrame | ✓ |

---

## (2) Accuracy on Unseen Data

Fresh CatBoost models (300 iterations, depth 5, lr 0.08) trained inside the test — no leakage from saved artefacts.

### Time Split: train Nov 2023 – Feb 2024 → test Mar – Apr 2024

| Metric | Value |
|--------|-------|
| Road-closure AUC (on future hold-out) | **0.8163** |
| Duration R² (on future hold-out) | −0.0445 |
| Duration MAE (on future hold-out) | 102.7 min |

Train: 5,576 rows · Test: 2,597 rows
Duration sub-set: 1,756 train / 777 test rows (rows where event resolved)

### Unseen-Corridor Holdout

Six corridors (`Mysore Road`, `Bellary Road 1`, `Hosur Road`, `ORR North 1`, `Tumkur Road`, `West of Chord Road`) withheld entirely from training. Model sees these corridors for the first time at inference.

| Metric | Value |
|--------|-------|
| Road-closure AUC on withheld corridors | **0.6963** |
| Withheld test rows | 2,558 |
| Closure events in test | 177 |

### Cold-Start Scenario

Rows within the corridor holdout whose **junction** was also never seen during training (doubly unseen geography).

| Metric | Value |
|--------|-------|
| Cold-start rows | **659** |
| Closure events | 56 |
| Road-closure AUC | **0.7312** |
| Probability range | [0.002, 0.949] — no crash, no out-of-range values |

---

## (3) Model Robustness — `predict_impact`

| Input | Duration ≥ 0 & finite | Closure prob ∈ [0,1] | No crash |
|-------|----------------------|----------------------|----------|
| Normal event (all fields) | ✓ (40.2 min) | ✓ (0.4046) | ✓ |
| All-unseen category values | ✓ (48.8 min) | ✓ (0.3605) | ✓ |
| Sparse — one key only | ✓ (38.7 min) | ✓ (0.3627) | ✓ |
| Empty dict `{}` | ✓ (36.8 min) | ✓ (0.5092) | ✓ |

---

## (4) Recommendation Robustness — `recommend()`

| Case | Status |
|------|--------|
| Low-impact event returns all 8 required keys | ✓ |
| High-impact event returns all 8 required keys | ✓ |
| High severity ≥ low severity (`High ≥ Low`) | ✓ |
| High officers ≥ low officers (10 ≥ 4) | ✓ |
| Closure-likely (prob 0.73) → `barricading = True` | ✓ |
| All 17 `event_cause` values in data → no crash | ✓ |
| Unseen corridor + unseen cause → valid output | ✓ |
| Empty dict `{}` → valid output | ✓ |

---

## (5) Dashboard Smoke Test — `AppTest`

| Step | Exceptions | Status |
|------|-----------|--------|
| `AppTest.from_file(app.py).run()` — initial load | 0 | ✓ |
| Click "Run Recommendation" form-submit button | 0 | ✓ |
| Output rendered (markdown widgets: 21, metric widgets: 8) | — | ✓ |

---

## Commentary

### What is strong

**Road-closure classification generalises well.**
AUC 0.8163 on the future time-split and 0.7312 on the doubly-unseen cold-start set show the model has learned genuine signal (event cause, corridor hotspot, priority, peak hour) rather than memorising training corridors. AUC 0.6963 on the six entirely withheld corridors is still clearly above chance (0.5), confirming the features transfer across unseen geographies.

**Graceful degradation.**
Every robustness scenario — empty dict, all-unseen categories, sparse input — returns a valid, in-range prediction without crashing. This is critical for a real-time BTP dispatch tool where field reporters may omit fields.

### What is weak (honest)

**Duration prediction is not predictive on unseen future data.**
R² of −0.0445 means the model explains none of the variance in future event duration: it performs worse than predicting the training-set mean for every event. The raw MAE of 102.7 min is heavily inflated by a long right tail (construction events averaging ~300 min, water-logging ~107 min, vehicle-breakdown ~41 min). Duration depends strongly on resources deployed *after* the event is reported — information unavailable at prediction time.

**Recommendation:** use the EDA cause-median benchmarks (`expected_clearance_min`) for field duration estimates, and treat the ML duration model as a directional signal only. The road-closure classifier is the operationally reliable output.
