"""Unit tests for the paired bootstrap equivalence (TOST) module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bio_is_curriculum.results.equivalence import (
    _macro_f1_mapped,
    bootstrap_equivalence,
    bootstrap_equivalence_folds,
    load_predictions,
)


def _balanced(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = np.repeat(np.arange(3), 100)  # 300 instances, 3 balanced classes
    y_pred = y_true.copy()
    return y_true, y_pred


def test_macro_f1_mapped_perfect_and_worst():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    assert _macro_f1_mapped(y_true, y_true, 3) == 1.0  # perfect
    # all wrong (shift by one class) -> 0 for every class
    assert _macro_f1_mapped(y_true, (y_true + 1) % 3, 3) == 0.0


def test_bootstrap_equivalence_identical_is_equivalent():
    y_true, y_pred = _balanced()
    res = bootstrap_equivalence(
        [(y_true, y_pred)],
        [(y_true, y_pred)],
        delta=0.01,
        n_bootstrap=500,
        seed=42,
    )
    assert res["delta_hat"] == 0.0
    assert res["ci_low"] == 0.0 and res["ci_high"] == 0.0
    assert res["equivalent"] is True


def test_bootstrap_equivalence_large_gap_not_equivalent():
    y_true, y_pred = _balanced()
    # IS+CL flips 30 correct -> wrong (drops macro-F1 from 1.0 to ~0.9)
    wrong = y_pred.copy()
    wrong[0:30] = (wrong[0:30] + 1) % 3
    res = bootstrap_equivalence(
        [(y_true, y_pred)],
        [(y_true, wrong)],
        delta=0.01,
        n_bootstrap=500,
        seed=42,
    )
    assert res["delta_hat"] < -0.05
    assert res["equivalent"] is False


def test_bootstrap_equivalence_detects_fold_mismatch():
    y_true, y_pred = _balanced()
    try:
        bootstrap_equivalence(
            [(y_true, y_pred)],
            [(y_true + 1, y_pred)],  # different y_true -> mismatch
            delta=0.01,
            n_bootstrap=10,
            seed=42,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_bootstrap_equivalence_folds_identical_equivalent():
    raw = np.array([0.80, 0.81, 0.79, 0.82, 0.80])
    res = bootstrap_equivalence_folds(raw, raw, delta=0.01, n_bootstrap=500, seed=42)
    assert res["delta_hat"] == 0.0
    assert res["ci_low"] == 0.0 and res["ci_high"] == 0.0
    assert res["equivalent"] is True
    assert res["n_folds"] == 5


def test_bootstrap_equivalence_folds_large_gap_not_equivalent():
    raw = np.array([0.80, 0.81, 0.79, 0.82, 0.80])
    iscl = raw - 0.10  # large, consistent drop across all folds
    res = bootstrap_equivalence_folds(raw, iscl, delta=0.01, n_bootstrap=500, seed=42)
    assert abs(res["delta_hat"] + 0.10) < 1e-9
    assert res["equivalent"] is False


def test_load_predictions_sorts_by_idx(tmp_path):
    path = tmp_path / "predictions_test.csv"
    df = pd.DataFrame(
        {
            "idx": [2, 0, 1],
            "y_true": [0, 1, 2],
            "y_pred": [1, 1, 0],
            "pred_entropy": [0.1, 0.2, 0.3],
        }
    )
    df.to_csv(path, index=False)
    loaded = load_predictions(path)
    assert list(loaded["idx"]) == [0, 1, 2]
    assert list(loaded["y_true"]) == [1, 2, 0]
