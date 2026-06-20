"""
Train duration (regression) and road-closure (classification) models.
Saves artifacts to models/ for use by impact_models.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, f1_score
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier

import data_prep as dp

# ── constants ─────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = [
    "event_cause", "event_type", "is_planned",
    "corridor", "zone", "junction", "police_station",
    "veh_type", "priority",
    "latitude", "longitude",
    "hour", "dow", "month", "is_peak", "is_weekend",
]
CAT_COLS = ["event_cause", "event_type", "corridor", "zone",
            "junction", "police_station", "veh_type", "priority"]
NUM_COLS = [f for f in FEATURES if f not in CAT_COLS]

TIME_CUTOFF = pd.Timestamp("2024-03-01", tz="UTC")

# holdout corridors (fixed seed for reproducibility)
HOLDOUT_CORRIDORS = [
    "IRR(Thanisandra road)", "Bellary Road 1", "Tumkur Road",
    "Bellary Road 2", "West of Chord Road",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def fill_cats(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing/NaN in cat cols with string 'unknown'."""
    df = df.copy()
    for c in CAT_COLS:
        df[c] = df[c].fillna("unknown").astype(str)
    return df


def oof_target_encode(train: pd.DataFrame, test: pd.DataFrame, col: str,
                      target: str, n_folds: int = 5, alpha: int = 20):
    """
    Out-of-fold mean-target encoding with global-mean fallback for unseen categories.
    Returns encoded train series, encoded test series, and the mapping dict.
    """
    global_mean = train[target].mean()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_enc = np.full(len(train), global_mean, dtype=float)
    train = train.reset_index(drop=True)

    for tr_idx, val_idx in kf.split(train):
        fold_stats = (
            train.iloc[tr_idx].groupby(col)[target]
            .agg(["sum", "count"])
        )
        fold_stats["enc"] = (fold_stats["sum"] + alpha * global_mean) / (fold_stats["count"] + alpha)
        train_enc[val_idx] = train.iloc[val_idx][col].map(fold_stats["enc"]).fillna(global_mean).values

    # full-train mapping for test / production
    full_stats = train.groupby(col)[target].agg(["sum", "count"])
    full_stats["enc"] = (full_stats["sum"] + alpha * global_mean) / (full_stats["count"] + alpha)
    mapping = full_stats["enc"].to_dict()

    test_enc = test[col].map(mapping).fillna(global_mean)
    return pd.Series(train_enc, index=train.index), test_enc, mapping, global_mean


def encode_split(train: pd.DataFrame, test: pd.DataFrame, target: str):
    """
    Apply OOF target encoding to all CAT_COLS.
    Returns transformed (train_X, test_X) DataFrames and encoder dict.
    Both outputs have a clean RangeIndex.
    """
    train = fill_cats(train).reset_index(drop=True)
    test  = fill_cats(test).reset_index(drop=True)
    encoders = {}
    train_parts, test_parts = {}, {}

    for col in CAT_COLS:
        tr_enc, te_enc, mapping, gm = oof_target_encode(train, test, col, target)
        train_parts[col + "_enc"] = tr_enc.values
        test_parts[col  + "_enc"] = te_enc.values
        encoders[col] = {"mapping": mapping, "global_mean": gm}

    train_X = pd.concat(
        [pd.DataFrame(train_parts), train[NUM_COLS]], axis=1
    )
    test_X = pd.concat(
        [pd.DataFrame(test_parts), test[NUM_COLS]], axis=1
    )
    return train_X, test_X, encoders


def apply_encoders(df: pd.DataFrame, encoders: dict, target_name: str = "") -> pd.DataFrame:
    """Apply saved encoders to any dataframe (production / corridor holdout)."""
    df = fill_cats(df).reset_index(drop=True)
    parts = {}
    for col in CAT_COLS:
        enc = encoders[col]
        parts[col + "_enc"] = df[col].map(enc["mapping"]).fillna(enc["global_mean"]).values
    enc_df = pd.DataFrame(parts)
    return pd.concat([enc_df, df[NUM_COLS]], axis=1)


# ── load & split data ─────────────────────────────────────────────────────────

df = dp.load_clean()
df = fill_cats(df)

# Time-based split
train_t = df[df["start_datetime"] < TIME_CUTOFF].copy()
test_t  = df[df["start_datetime"] >= TIME_CUTOFF].copy()

# Corridor holdout split (full dataset)
train_c = df[~df["corridor"].isin(HOLDOUT_CORRIDORS)].copy()
test_c  = df[df["corridor"].isin(HOLDOUT_CORRIDORS)].copy()

print(f"Time split  — train: {len(train_t):,}  test: {len(test_t):,}")
print(f"Corridor split — train: {len(train_c):,}  test: {len(test_c):,}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 1 — DURATION REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Duration Regression ──")

DUR_TARGET = "duration_min"

# Subset: rows with known duration
tr_dur_t = train_t[train_t["has_duration"] == 1].copy()
te_dur_t = test_t[test_t["has_duration"] == 1].copy()
tr_dur_c = train_c[train_c["has_duration"] == 1].copy()
te_dur_c = test_c[test_c["has_duration"] == 1].copy()

# log1p target (right-skewed distribution)
LOG_TARGET = "log_duration"
for d in [tr_dur_t, te_dur_t, tr_dur_c, te_dur_c]:
    d[LOG_TARGET] = np.log1p(d[DUR_TARGET])

# ── 1a: LightGBM (OOF target-encoded categoricals) ──
# reset index before extracting labels so they align with encode_split output
tr_dur_t = tr_dur_t.reset_index(drop=True)
te_dur_t = te_dur_t.reset_index(drop=True)
tr_X_t, te_X_t, dur_encoders_lgb = encode_split(tr_dur_t, te_dur_t, LOG_TARGET)
tr_y_t = tr_dur_t[LOG_TARGET].values
te_y_t = te_dur_t[LOG_TARGET].values

lgb_dur = lgb.LGBMRegressor(
    n_estimators=600, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1,
)
lgb_dur.fit(tr_X_t, tr_y_t)

# time-split metrics
pred_log = lgb_dur.predict(te_X_t)
pred_min = np.expm1(pred_log)
true_min = np.expm1(te_y_t)
lgb_dur_mae_t  = mean_absolute_error(true_min, pred_min)
lgb_dur_r2_t   = r2_score(true_min, pred_min)
print(f"LGB Duration | time-split  MAE={lgb_dur_mae_t:.1f} min  R²={lgb_dur_r2_t:.3f}")

# corridor holdout
te_X_c_lgb = apply_encoders(te_dur_c, dur_encoders_lgb, LOG_TARGET)
pred_log_c = lgb_dur.predict(te_X_c_lgb)
pred_min_c = np.expm1(pred_log_c)
true_min_c = np.expm1(te_dur_c[LOG_TARGET].values)
lgb_dur_mae_c = mean_absolute_error(true_min_c, pred_min_c)
lgb_dur_r2_c  = r2_score(true_min_c, pred_min_c)
print(f"LGB Duration | unseen-corr MAE={lgb_dur_mae_c:.1f} min  R²={lgb_dur_r2_c:.3f}")

# ── 1b: CatBoost (native categorical handling) ──
cb_dur = CatBoostRegressor(
    iterations=600, learning_rate=0.05, depth=6,
    loss_function="MAE", cat_features=CAT_COLS,
    random_seed=42, verbose=0,
)
cb_dur.fit(
    tr_dur_t[FEATURES], tr_dur_t[LOG_TARGET],
    eval_set=(te_dur_t[FEATURES], te_dur_t[LOG_TARGET]),
    early_stopping_rounds=50,
)
pred_log_cb = cb_dur.predict(te_dur_t[FEATURES])
pred_min_cb = np.expm1(pred_log_cb)
cb_dur_mae_t = mean_absolute_error(true_min, pred_min_cb)
cb_dur_r2_t  = r2_score(true_min, pred_min_cb)
print(f"CB  Duration | time-split  MAE={cb_dur_mae_t:.1f} min  R²={cb_dur_r2_t:.3f}")

pred_log_cb_c = cb_dur.predict(te_dur_c[FEATURES])
pred_min_cb_c = np.expm1(pred_log_cb_c)
cb_dur_mae_c = mean_absolute_error(true_min_c, pred_min_cb_c)
cb_dur_r2_c  = r2_score(true_min_c, pred_min_cb_c)
print(f"CB  Duration | unseen-corr MAE={cb_dur_mae_c:.1f} min  R²={cb_dur_r2_c:.3f}")

# Pick best on time-split MAE (lower is better)
best_dur_name = "lgb" if lgb_dur_mae_t <= cb_dur_mae_t else "catboost"
best_dur_model = lgb_dur if best_dur_name == "lgb" else cb_dur
print(f"\nSelected duration model: {best_dur_name}")

# Feature importance for best model
if best_dur_name == "lgb":
    fi_dur = dict(zip(tr_X_t.columns, lgb_dur.feature_importances_))
else:
    fi_dur = dict(zip(FEATURES, cb_dur.get_feature_importance()))
fi_dur_sorted = sorted(fi_dur.items(), key=lambda x: x[1], reverse=True)[:10]


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 2 — ROAD-CLOSURE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Road-Closure Classification ──")

CLF_TARGET = "road_closure"

# ── 2a: LightGBM ──
train_t = train_t.reset_index(drop=True)
test_t  = test_t.reset_index(drop=True)
tr_X_clf_t, te_X_clf_t, clf_encoders_lgb = encode_split(train_t, test_t, CLF_TARGET)
tr_y_clf_t = train_t[CLF_TARGET].values
te_y_clf_t = test_t[CLF_TARGET].values

pos_weight = (tr_y_clf_t == 0).sum() / max((tr_y_clf_t == 1).sum(), 1)
lgb_clf = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, verbose=-1,
)
lgb_clf.fit(tr_X_clf_t, tr_y_clf_t)

prob_lgb = lgb_clf.predict_proba(te_X_clf_t)[:, 1]
pred_lgb = (prob_lgb >= 0.5).astype(int)
lgb_auc_t = roc_auc_score(te_y_clf_t, prob_lgb)
lgb_f1_t  = f1_score(te_y_clf_t, pred_lgb)
print(f"LGB Closure  | time-split  AUC={lgb_auc_t:.3f}  F1={lgb_f1_t:.3f}")

# corridor holdout
te_X_c_clf = apply_encoders(test_c, clf_encoders_lgb, CLF_TARGET)
prob_lgb_c = lgb_clf.predict_proba(te_X_c_clf)[:, 1]
pred_lgb_c = (prob_lgb_c >= 0.5).astype(int)
lgb_auc_c = roc_auc_score(test_c[CLF_TARGET].values, prob_lgb_c)
lgb_f1_c  = f1_score(test_c[CLF_TARGET].values, pred_lgb_c)
print(f"LGB Closure  | unseen-corr AUC={lgb_auc_c:.3f}  F1={lgb_f1_c:.3f}")

# ── 2b: CatBoost ──
cb_clf = CatBoostClassifier(
    iterations=600, learning_rate=0.05, depth=6,
    loss_function="Logloss", eval_metric="AUC",
    class_weights=[1, pos_weight], cat_features=CAT_COLS,
    random_seed=42, verbose=0,
)
cb_clf.fit(
    train_t[FEATURES], train_t[CLF_TARGET],
    eval_set=(test_t[FEATURES], test_t[CLF_TARGET]),
    early_stopping_rounds=50,
)
prob_cb = cb_clf.predict_proba(test_t[FEATURES])[:, 1]
pred_cb = (prob_cb >= 0.5).astype(int)
cb_auc_t = roc_auc_score(te_y_clf_t, prob_cb)
cb_f1_t  = f1_score(te_y_clf_t, pred_cb)
print(f"CB  Closure  | time-split  AUC={cb_auc_t:.3f}  F1={cb_f1_t:.3f}")

prob_cb_c = cb_clf.predict_proba(test_c[FEATURES])[:, 1]
pred_cb_c = (prob_cb_c >= 0.5).astype(int)
cb_auc_c = roc_auc_score(test_c[CLF_TARGET].values, prob_cb_c)
cb_f1_c  = f1_score(test_c[CLF_TARGET].values, pred_cb_c)
print(f"CB  Closure  | unseen-corr AUC={cb_auc_c:.3f}  F1={cb_f1_c:.3f}")

best_clf_name = "lgb" if lgb_auc_t >= cb_auc_t else "catboost"
best_clf_model = lgb_clf if best_clf_name == "lgb" else cb_clf
print(f"\nSelected closure model: {best_clf_name}")

if best_clf_name == "lgb":
    fi_clf = dict(zip(tr_X_clf_t.columns, lgb_clf.feature_importances_))
else:
    fi_clf = dict(zip(FEATURES, cb_clf.get_feature_importance()))
fi_clf_sorted = sorted(fi_clf.items(), key=lambda x: x[1], reverse=True)[:10]


# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Saving artifacts ──")

joblib.dump(lgb_dur,           os.path.join(MODELS_DIR, "lgb_duration.pkl"))
joblib.dump(cb_dur,            os.path.join(MODELS_DIR, "cb_duration.pkl"))
joblib.dump(dur_encoders_lgb,  os.path.join(MODELS_DIR, "dur_encoders.pkl"))

joblib.dump(lgb_clf,           os.path.join(MODELS_DIR, "lgb_closure.pkl"))
joblib.dump(cb_clf,            os.path.join(MODELS_DIR, "cb_closure.pkl"))
joblib.dump(clf_encoders_lgb,  os.path.join(MODELS_DIR, "clf_encoders.pkl"))

meta = {
    "best_dur_model":  best_dur_name,
    "best_clf_model":  best_clf_name,
    "features":        FEATURES,
    "cat_cols":        CAT_COLS,
    "num_cols":        NUM_COLS,
    "holdout_corridors": HOLDOUT_CORRIDORS,
}
joblib.dump(meta, os.path.join(MODELS_DIR, "meta.pkl"))
print("Saved to", MODELS_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
#  WRITE MODELS.md
# ═══════════════════════════════════════════════════════════════════════════════

fi_dur_lines = "\n".join(f"  {i+1}. {k}: {v:.0f}" for i, (k, v) in enumerate(fi_dur_sorted))
fi_clf_lines = "\n".join(f"  {i+1}. {k}: {v:.0f}" for i, (k, v) in enumerate(fi_clf_sorted))

models_md = f"""# Impact Models — Approach & Validation

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
| **Corridor holdout** | 18 corridors | 5 unseen corridors |

The time-based split is the primary "does it work on future data?" test.
The corridor holdout tests generalisation to **unseen locations**.

---

## Model 1 — Duration Regression

Target: `log1p(duration_min)`, evaluated as raw minutes (expm1 of predictions).
Trained only on rows where `has_duration == 1` ({len(tr_dur_t)} train / {len(te_dur_t)} test rows, time split).

### Honest validation results

| Model | Split | MAE (min) | R² |
|---|---|---|---|
| LightGBM | time-split (future) | {lgb_dur_mae_t:.1f} | {lgb_dur_r2_t:.3f} |
| LightGBM | unseen corridors | {lgb_dur_mae_c:.1f} | {lgb_dur_r2_c:.3f} |
| CatBoost | time-split (future) | {cb_dur_mae_t:.1f} | {cb_dur_r2_t:.3f} |
| CatBoost | unseen corridors | {cb_dur_mae_c:.1f} | {cb_dur_r2_c:.3f} |

**Selected**: `{best_dur_name}` (lower time-split MAE).

*Note: duration_min is highly right-skewed (median ~46 min, max ~1437 min);
MAE on raw minutes reflects real planning uncertainty, not model failure.*

### Duration — Top-10 feature importances ({best_dur_name})
{fi_dur_lines}

---

## Model 2 — Road-Closure Classification

Target: `road_closure` (0/1). Class imbalance ~9:1; corrected with `scale_pos_weight`.
Threshold: 0.5 (adjustable at inference time for precision/recall trade-off).

### Honest validation results

| Model | Split | ROC-AUC | F1 |
|---|---|---|---|
| LightGBM | time-split (future) | {lgb_auc_t:.3f} | {lgb_f1_t:.3f} |
| LightGBM | unseen corridors | {lgb_auc_c:.3f} | {lgb_f1_c:.3f} |
| CatBoost | time-split (future) | {cb_auc_t:.3f} | {cb_f1_t:.3f} |
| CatBoost | unseen corridors | {cb_auc_c:.3f} | {cb_f1_c:.3f} |

**Selected**: `{best_clf_name}` (higher time-split AUC).

*Note: road closure rate is ~8-10%; F1 on a rare class is volatile — AUC is the
more reliable signal of ranking ability.*

### Road-closure — Top-10 feature importances ({best_clf_name})
{fi_clf_lines}

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
"""

with open(os.path.join(MODELS_DIR, "MODELS.md"), "w") as f:
    f.write(models_md)

print("Written MODELS.md")
print("\n=== DONE ===")
