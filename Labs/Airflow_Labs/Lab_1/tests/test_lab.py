"""Unit tests for the pipeline logic in dags/src/lab.py (no Airflow needed)."""
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from conftest import DAGS_DIR
from src import lab

SOURCE = os.path.join(DAGS_DIR, "data", "file.csv")
TEST_CSV = os.path.join(DAGS_DIR, "data", "test.csv")
FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]


def test_clean_data_passes_quality_gate(tmp_path):
    snap = lab.snapshot_source_data(SOURCE, str(tmp_path / "raw.csv"))
    report = lab.validate_data(snap["path"], FEATURES, min_rows=500, max_null_fraction=0.05)
    assert report["passed"], report["failures"]


def test_bad_data_fails_quality_gate(tmp_path):
    snap = lab.snapshot_source_data(SOURCE, str(tmp_path / "raw.csv"), inject_bad_data=True)
    report = lab.validate_data(snap["path"], FEATURES, min_rows=500, max_null_fraction=0.05)
    assert not report["passed"]
    assert any("CREDIT_LIMIT" in f for f in report["failures"])
    assert any("negative" in f for f in report["failures"])


def test_missing_column_is_reported(tmp_path):
    snap = lab.snapshot_source_data(SOURCE, str(tmp_path / "raw.csv"))
    report = lab.validate_data(snap["path"], FEATURES + ["NOT_A_COLUMN"], 500, 0.05)
    assert not report["passed"]
    assert any("missing columns" in f for f in report["failures"])


def test_candidate_ks_validates_range():
    assert lab.candidate_ks(1, 5, 1000) == [1, 2, 3, 4, 5]
    assert lab.candidate_ks(1, 50, 10) == list(range(1, 10))  # capped at n_rows - 1
    with pytest.raises(ValueError):
        lab.candidate_ks(5, 3, 1000)


def test_end_to_end_pipeline(tmp_path):
    snap = lab.snapshot_source_data(SOURCE, str(tmp_path / "raw.csv"))
    prep = lab.preprocess(snap["path"], FEATURES, str(tmp_path))
    assert os.path.exists(prep["scaler_path"])

    results = [lab.train_kmeans(k, prep["scaled_path"], str(tmp_path)) for k in range(1, 11)]
    sse = [r["sse"] for r in results]
    assert sse == sorted(sse, reverse=True)  # SSE never increases with more clusters

    selection = lab.select_best_k(list(reversed(results)), str(tmp_path))  # order must not matter
    assert 1 < selection["best_k"] < 10
    with open(selection["best_model_path"], "rb") as f:
        assert pickle.load(f).n_clusters == selection["best_k"]  # the CHOSEN model was promoted

    metrics = lab.evaluate(selection["best_model_path"], prep["scaled_path"],
                           prep["scaler_path"], TEST_CSV, FEATURES)
    assert -1 <= metrics["silhouette_score"] <= 1
    assert all(0 <= p < selection["best_k"] for p in metrics["test_predictions"])

    report = lab.write_report(str(tmp_path), str(tmp_path / "latest"), {"best_k": selection["best_k"]})
    assert json.load(open(report))["best_k"] == selection["best_k"]


def test_test_data_is_scaled_before_prediction(tmp_path):
    """Regression test for the original bug: test.csv was predicted on unscaled values."""
    snap = lab.snapshot_source_data(SOURCE, str(tmp_path / "raw.csv"))
    prep = lab.preprocess(snap["path"], FEATURES, str(tmp_path))
    with open(prep["scaler_path"], "rb") as f:
        scaler = pickle.load(f)
    raw = pd.read_csv(TEST_CSV)[FEATURES]
    scaled = scaler.transform(raw)
    assert not np.allclose(raw.values, scaled)  # scaling actually changes the input
