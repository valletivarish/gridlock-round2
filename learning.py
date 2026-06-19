"""
Pillar 3 — Post-Event Learning / Drift-Adaptation Demo
=======================================================
Demonstrates that periodic retraining keeps the road-closure classifier
accurate as real distributional drift unfolds across months.

Approach
--------
Data runs Nov 2023 → Apr 2024.  We treat each calendar month (Jan–Apr 2024)
as a test window and compare two strategies:

  STATIC    — trained once on Nov+Dec 2023 only, frozen for all future months.
  RETRAINED — retrained cumulatively on all data before the test month.

We report ROC-AUC per month for both strategies and plot them side by side.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_PATH = os.path.join(HERE, "eda", "fig_07_learning.png")

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
CAT_FEATS = ["event_cause", "corridor", "zone", "junction",
             "police_station", "veh_type", "priority"]
NUM_FEATS = ["is_planned", "latitude", "longitude",
             "hour", "dow", "month", "is_peak", "is_weekend"]
ALL_FEATS = CAT_FEATS + NUM_FEATS
TARGET    = "road_closure"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_period_key(row):
    """Return a sortable (year, month) tuple for each row."""
    return (row["start_datetime"].year, row["month"])


def add_period(df):
    df = df.copy()
    df["_year"]   = df["start_datetime"].dt.year
    df["_period"] = list(zip(df["_year"], df["month"]))
    return df


def make_xy(df):
    X = df[ALL_FEATS].copy()
    # CatBoost needs string categoricals; fill any remaining NaN just in case
    for c in CAT_FEATS:
        X[c] = X[c].astype(str).fillna("unknown")
    for c in NUM_FEATS:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    y = df[TARGET].values
    return X, y


def fit_catboost(X_train, y_train):
    cat_idx = [X_train.columns.get_loc(c) for c in CAT_FEATS]
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=5,
        eval_metric="AUC",
        random_seed=42,
        verbose=0,
    )
    model.fit(X_train, y_train, cat_features=cat_idx)
    return model


def evaluate(model, X_test, y_test):
    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        return float("nan")          # AUC undefined if only one class present
    probs = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, probs)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(df):
    """
    Returns a DataFrame with columns:
        period_label, n_train, n_test, pos_rate, auc_static, auc_retrained
    """
    df = add_period(df)

    # Ordered list of all (year, month) periods present in the data
    all_periods = sorted(df["_period"].unique())
    # periods used as test windows: Jan-2024 through Apr-2024
    test_periods = [(y, m) for y, m in all_periods if y == 2024]

    # STATIC model — trained on Nov+Dec 2023 only (earliest two full months)
    static_train_mask = df["_period"].isin([(2023, 11), (2023, 12)])
    X_static, y_static = make_xy(df[static_train_mask])
    print(f"Static train size: {len(y_static)}  (positives: {y_static.sum()})")
    static_model = fit_catboost(X_static, y_static)

    rows = []
    for period in test_periods:
        year, month = period
        label = f"{year}-{month:02d}"

        # --- test set: current month ---
        test_mask  = df["_period"] == period
        X_test, y_test = make_xy(df[test_mask])

        # --- cumulative training set: everything strictly before this month ---
        train_mask = df["_period"] < period
        X_train, y_train = make_xy(df[train_mask])

        print(f"\n{label}  |  train={train_mask.sum()}  test={test_mask.sum()}"
              f"  pos_in_test={y_test.sum()}")

        # Retrained model on cumulative data
        retrained_model = fit_catboost(X_train, y_train)

        auc_static    = evaluate(static_model,    X_test, y_test)
        auc_retrained = evaluate(retrained_model, X_test, y_test)

        rows.append({
            "period_label":  label,
            "n_train":       train_mask.sum(),
            "n_test":        test_mask.sum(),
            "pos_rate":      round(y_test.mean(), 4),
            "auc_static":    round(auc_static,    4),
            "auc_retrained": round(auc_retrained, 4),
        })
        print(f"  AUC  static={auc_static:.4f}   retrained={auc_retrained:.4f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def plot_results(results, path):
    fig, ax = plt.subplots(figsize=(8, 5))

    x     = np.arange(len(results))
    width = 0.35

    bars_s = ax.bar(x - width/2, results["auc_static"],    width,
                    label="Static (trained Nov–Dec 2023)", color="#e07b54", alpha=0.85)
    bars_r = ax.bar(x + width/2, results["auc_retrained"], width,
                    label="Retrained (cumulative)",         color="#4c8bbf", alpha=0.85)

    # Annotate bar tops
    for bar in bars_s:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_r:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(results["period_label"])
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Test Month")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Road-Closure Classifier: Static vs Retrained\n(Pillar 3 — Post-Event Learning / Drift Adaptation)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(HERE, ".."))
    from round_2.data_prep import load_clean

    print("Loading data …")
    df = load_clean()
    print(f"Dataset: {len(df)} rows, {df['road_closure'].mean():.2%} road-closure rate")

    results = run_experiment(df)

    print("\n=== Results ===")
    print(results.to_string(index=False))

    plot_results(results, FIG_PATH)

    # Persist results table for the markdown report
    results.to_csv(os.path.join(HERE, "eda", "table_learning_results.csv"), index=False)
