"""Data-quality, preprocessing, K-Means training and evaluation functions.

This module has no Airflow imports, so every function can be unit-tested on its
own (see tests/test_lab.py). The DAG tasks are thin wrappers around these.

Tasks exchange FILE PATHS and small JSON-safe dicts through XCom, never
DataFrames or models; the heavy artifacts live in the shared working_data folder.
"""
import json
import os
import pickle
import shutil

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def snapshot_source_data(source_csv, snapshot_csv, inject_bad_data=False, seed=42):
    """Copy the source dataset into the landing zone, optionally corrupting it.

    inject_bad_data=True blanks 20% of CREDIT_LIMIT and makes some PURCHASES
    negative, so the training DAG's data-quality branch can be demonstrated.
    """
    df = pd.read_csv(source_csv)
    if inject_bad_data:
        rng = np.random.default_rng(seed)
        null_idx = rng.choice(df.index, size=int(len(df) * 0.20), replace=False)
        df.loc[null_idx, "CREDIT_LIMIT"] = np.nan
        neg_idx = rng.choice(df.index, size=50, replace=False)
        df.loc[neg_idx, "PURCHASES"] = -df.loc[neg_idx, "PURCHASES"].abs() - 1
    os.makedirs(os.path.dirname(snapshot_csv), exist_ok=True)
    tmp = snapshot_csv + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, snapshot_csv)  # atomic: consumers never read a half-written file
    return {"path": snapshot_csv, "rows": int(len(df)), "bad_data_injected": bool(inject_bad_data)}


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
def validate_data(csv_path, features, min_rows, max_null_fraction):
    """Run data-quality checks and return a JSON-safe report with passed=True/False."""
    df = pd.read_csv(csv_path)
    failures = []

    missing = [c for c in features if c not in df.columns]
    if missing:
        failures.append(f"missing columns: {missing}")
    present = [c for c in features if c in df.columns]

    if len(df) < min_rows:
        failures.append(f"only {len(df)} rows, need at least {min_rows}")

    null_fraction = {c: round(float(df[c].isna().mean()), 4) for c in present}
    for col, frac in null_fraction.items():
        if frac > max_null_fraction:
            failures.append(f"{col}: {frac:.1%} nulls exceeds limit of {max_null_fraction:.1%}")

    negatives = {c: int((df[c] < 0).sum()) for c in present}
    for col, count in negatives.items():
        if count > 0:
            failures.append(f"{col}: {count} negative values (expected >= 0)")

    return {
        "passed": not failures,
        "rows": int(len(df)),
        "features": list(features),
        "null_fraction": null_fraction,
        "negative_values": negatives,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(csv_path, features, out_dir):
    """Drop rows with missing features, fit a MinMaxScaler, and SAVE the scaler.

    The original lab never saved the scaler, so test data was predicted on raw,
    unscaled values. The saved scaler is applied to the test data in evaluate().
    """
    from sklearn.preprocessing import MinMaxScaler

    df = pd.read_csv(csv_path)
    data = df[features].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    os.makedirs(out_dir, exist_ok=True)
    scaled_path = os.path.join(out_dir, "scaled_features.npy")
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    np.save(scaled_path, scaled)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    return {
        "scaled_path": scaled_path,
        "scaler_path": scaler_path,
        "features": list(features),
        "rows_in": int(len(df)),
        "rows_used": int(len(data)),
    }


def candidate_ks(k_min, k_max, n_rows):
    """List of cluster counts to try; k can't exceed the number of rows."""
    if k_min < 1 or k_max <= k_min:
        raise ValueError(f"invalid k range: k_min={k_min}, k_max={k_max} (need 1 <= k_min < k_max)")
    k_max = min(k_max, n_rows - 1)
    return list(range(k_min, k_max + 1))


# ---------------------------------------------------------------------------
# Training (one call per k, run in parallel by dynamic task mapping)
# ---------------------------------------------------------------------------
def train_kmeans(k, scaled_path, out_dir, seed=42):
    """Fit one K-Means model for a single k, save it, and return its SSE."""
    from sklearn.cluster import KMeans

    X = np.load(scaled_path)
    model = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300, random_state=seed)
    model.fit(X)

    model_dir = os.path.join(out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"kmeans_k{k:02d}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return {"k": int(k), "sse": float(model.inertia_), "model_path": model_path}


def find_elbow(ks, sse):
    """Elbow via kneed's KneeLocator, falling back to max curvature if none is found."""
    from kneed import KneeLocator

    knee = KneeLocator(ks, sse, curve="convex", direction="decreasing").elbow
    if knee is not None:
        return int(knee), "kneed"
    if len(ks) >= 3:
        second_diff = np.diff(sse, n=2)
        return int(ks[int(np.argmax(second_diff)) + 1]), "max_curvature_fallback"
    return int(ks[0]), "single_candidate"


def select_best_k(results, out_dir):
    """Pick the elbow k from all mapped results and promote that model.

    The original lab saved whichever model was fitted LAST (k=49) instead of
    the k the elbow method actually chose. Here the chosen model is promoted.
    """
    results = sorted(results, key=lambda r: r["k"])
    ks = [r["k"] for r in results]
    sse = [r["sse"] for r in results]
    best_k, method = find_elbow(ks, sse)

    best = next(r for r in results if r["k"] == best_k)
    best_model_path = os.path.join(out_dir, "best_model.pkl")
    shutil.copyfile(best["model_path"], best_model_path)

    plot_path = _plot_elbow(ks, sse, best_k, out_dir)
    return {
        "best_k": best_k,
        "elbow_method": method,
        "best_model_path": best_model_path,
        "sse_by_k": {str(k): round(s, 4) for k, s in zip(ks, sse)},
        "elbow_plot": plot_path,
    }


def _plot_elbow(ks, sse, best_k, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, sse, marker="o")
    ax.axvline(best_k, color="red", linestyle="--", label=f"elbow: k={best_k}")
    ax.set_xlabel("number of clusters (k)")
    ax.set_ylabel("SSE (inertia)")
    ax.set_title("Elbow method")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "elbow_plot.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Evaluation and reporting
# ---------------------------------------------------------------------------
def evaluate(best_model_path, scaled_path, scaler_path, test_csv, features, seed=42):
    """Silhouette score on training data + cluster predictions for test.csv (scaled correctly)."""
    from sklearn.metrics import silhouette_score

    with open(best_model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    X = np.load(scaled_path)
    labels = model.predict(X)
    n_clusters = len(set(labels))
    silhouette = None
    if 1 < n_clusters < len(X):
        silhouette = float(silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=seed))

    test_df = pd.read_csv(test_csv)
    test_scaled = scaler.transform(test_df[features])  # the fix: scale test data like training data
    test_pred = model.predict(test_scaled)

    sizes = pd.Series(labels).value_counts().sort_index()
    return {
        "silhouette_score": None if silhouette is None else round(silhouette, 4),
        "cluster_sizes": {str(int(c)): int(n) for c, n in sizes.items()},
        "test_predictions": [int(p) for p in test_pred],
    }


def write_report(out_dir, latest_dir, summary):
    """Write run_summary.json for this run and copy it (plus the plot) to latest/."""
    os.makedirs(latest_dir, exist_ok=True)
    path = os.path.join(out_dir, "run_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    shutil.copyfile(path, os.path.join(latest_dir, "run_summary.json"))
    plot = os.path.join(out_dir, "elbow_plot.png")
    if os.path.exists(plot):
        shutil.copyfile(plot, os.path.join(latest_dir, "elbow_plot.png"))
    return path
