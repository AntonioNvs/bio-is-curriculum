"""Unit tests for result discovery and fold-paired significance tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bio_is_curriculum.results.aggregator import (
    collect_per_fold_values,
    reduce_phase_metric,
)
from bio_is_curriculum.results.significance import holm_adjust, paired_tests


def _phase_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_reduce_phase_metric_max_last_sum():
    df = _phase_df(
        [
            {"phase": "clean", "macro_f1": 0.70, "train_time_s": 10.0, "steps_to_best_val": 5},
            {"phase": "hard", "macro_f1": 0.75, "train_time_s": 20.0, "steps_to_best_val": 9},
        ]
    )
    assert reduce_phase_metric(df, "macro_f1") == 0.75  # max over phases
    assert reduce_phase_metric(df, "steps_to_best_val") == 9  # last phase
    assert reduce_phase_metric(df, "train_time_s") == 30.0  # sum over phases


def test_collect_per_fold_values_skips_missing_and_nan(tmp_path):
    mode = "raw"
    for fold in (0, 1, 3):  # fold2 intentionally missing
        fold_dir = tmp_path / f"{mode}_fold{fold}"
        fold_dir.mkdir(parents=True)
        _phase_df([{"phase": "full", "macro_f1": 0.8 + fold * 0.01}]).to_csv(
            fold_dir / "phase_metrics.csv", index=False
        )
    # fold3 holds NaN macro_f1 -> skipped
    fold_dir = tmp_path / f"{mode}_fold4"
    fold_dir.mkdir(parents=True)
    _phase_df([{"phase": "full", "macro_f1": float("nan")}]).to_csv(
        fold_dir / "phase_metrics.csv", index=False
    )

    vals = collect_per_fold_values(str(tmp_path), mode, list(range(5)), "macro_f1")
    assert np.allclose(vals, [0.80, 0.81, 0.83])  # folds 0,1,3 in ascending order


def test_holm_adjust_known_case():
    p = np.array([0.01, 0.02, 0.03, 0.5])
    adjusted = holm_adjust(p)
    assert np.allclose(adjusted, [0.04, 0.06, 0.06, 0.5])


def test_holm_adjust_treats_nan_as_non_significant():
    p = np.array([0.01, np.nan])
    adjusted = holm_adjust(p)
    assert adjusted[0] == 0.02
    assert adjusted[1] == 1.0


def test_paired_tests_holm_and_nan_pair():
    values = {
        "A": {0: 0.90, 1: 0.80, 2: 0.85},
        "B": {0: 0.70, 1: 0.60, 2: 0.65},
    }
    rows = paired_tests(values, table="factorial", dataset="webkb", alpha=0.05)
    assert len(rows) == 1
    row = rows[0]
    assert row["n_folds"] == 3
    assert np.isclose(row["mean_a"], 0.85)
    assert np.isclose(row["mean_b"], 0.65)
    assert np.isclose(row["delta"], 0.20)
    assert row["p_ttest_holm"] == row["p_ttest"]  # single pair: no correction
    assert row["sig_ttest_holm"] is True


def test_paired_tests_with_missing_fold_aligns_intersection():
    values = {
        "A": {0: 0.90, 1: 0.80, 2: 0.85},
        "B": {0: 0.70, 2: 0.65},  # missing fold 1
    }
    rows = paired_tests(values, table="factorial", dataset="webkb", alpha=0.05)
    assert rows[0]["n_folds"] == 2  # intersection of {0,2}


def test_paired_tests_pairs_filter():
    values = {
        "A": {0: 0.90, 1: 0.80, 2: 0.85},
        "B": {0: 0.70, 1: 0.60, 2: 0.65},
        "C": {0: 0.88, 1: 0.79, 2: 0.84},
    }
    rows = paired_tests(
        values,
        table="factorial",
        dataset="webkb",
        alpha=0.05,
        pairs={frozenset({"A", "B"})},
    )
    assert len(rows) == 1
    assert {rows[0]["method_a"], rows[0]["method_b"]} == {"A", "B"}
