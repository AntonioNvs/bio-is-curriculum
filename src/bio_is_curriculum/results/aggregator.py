"""Cross-fold experiment aggregation."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

_MAX_METRICS = {
    "micro_f1", "macro_f1", "f1_weighted", "accuracy",
    "hard_slice_macro_f1", "best_val_macro_f1",
}
_LAST_METRICS = {
    "avg_seq_len", "compute_proxy", "steps_to_best_val", "n_train_instances",
}
_SUM_METRICS = {"train_time_s", "pred_time_s"}


def reduce_phase_metric(df: pd.DataFrame, metric: str) -> float:
    """Collapse a fold's ``phase_metrics.csv`` rows into a single value.

    Mirrors the per-fold reduction used to build ``summary.csv``: maximum over
    phases for performance metrics, last phase for training-state metrics, and
    sum over phases for timings.
    """
    if metric in _MAX_METRICS:
        return float(df[metric].max())
    if metric in _LAST_METRICS:
        return float(df[metric].iloc[-1])
    if metric in _SUM_METRICS:
        return float(df[metric].sum())
    return float("nan")


def collect_per_fold_map(
    experiment_dir: str,
    mode: str,
    folds: list[int],
    metric: str,
) -> dict[int, float]:
    """Return ``{fold_index: metric_value}`` for one mode/metric.

    Missing/empty files and NaN values are skipped, so the returned mapping only
    contains folds with a valid value (needed for fold-paired statistical tests).
    """
    result: dict[int, float] = {}
    for fold in folds:
        path = os.path.join(experiment_dir, f"{mode}_fold{fold}", "phase_metrics.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty or metric not in df.columns or len(df[metric]) == 0:
            continue
        val = reduce_phase_metric(df, metric)
        if np.isnan(val):
            continue
        result[fold] = val
    return result


def collect_per_fold_values(
    experiment_dir: str,
    mode: str,
    folds: list[int],
    metric: str,
) -> list[float]:
    """Return per-fold metric values in ascending fold order."""
    per_fold = collect_per_fold_map(experiment_dir, mode, folds, metric)
    return [per_fold[f] for f in sorted(per_fold)]


def aggregate(experiment_dir: str, modes: list[str], folds: list[int]) -> pd.DataFrame:
    metrics_of_interest = list(_MAX_METRICS | _LAST_METRICS | _SUM_METRICS)
    records: dict[str, dict[str, list[float]]] = {
        m: {k: [] for k in metrics_of_interest} for m in modes
    }
    phase_counts: dict[str, list[int]] = {m: [] for m in modes}
    missing: list[str] = []

    for mode in modes:
        for fold in folds:
            path = os.path.join(experiment_dir, f"{mode}_fold{fold}", "phase_metrics.csv")
            if not os.path.exists(path):
                missing.append(path)
                continue
            df = pd.read_csv(path)
            if df.empty:
                missing.append(path)
                continue
            phase_counts[mode].append(len(df))

            for metric in metrics_of_interest:
                if metric in df.columns and len(df[metric]) > 0:
                    val = reduce_phase_metric(df, metric)
                    if not np.isnan(val):
                        records[mode][metric].append(val)

    if missing:
        print(f"\nWARNING: {len(missing)} missing/empty result file(s).")

    reduction_by_mode: dict[str, float] = {}
    n_removed_by_mode: dict[str, list[float]] = {m: [] for m in modes}
    for mode in modes:
        for fold in folds:
            is_path = os.path.join(experiment_dir, f"{mode}_fold{fold}", "instance_selection.json")
            if os.path.exists(is_path):
                with open(is_path, encoding="utf-8") as f:
                    is_data = json.load(f)
                reduction = float(is_data.get("reduction", 0.0))
                n_removed = float(is_data.get("n_removed", 0.0))
                reduction_by_mode.setdefault(mode, reduction)
                n_removed_by_mode[mode].append(n_removed)

    rows = []
    for mode in modes:
        for metric in metrics_of_interest:
            vals = np.array(records[mode][metric])
            n = len(vals)
            if n == 0:
                rows.append(_empty_row(mode, metric))
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
            if n > 1:
                margin = float(stats.t.ppf(0.975, df=n - 1)) * std / np.sqrt(n)
                ci_low, ci_high = mean - margin, mean + margin
            else:
                ci_low = ci_high = float("nan")
            rows.append({
                "mode": mode, "metric": metric,
                "mean": mean, "std": std,
                "ci_95_low": ci_low, "ci_95_high": ci_high,
                "n_folds": n,
            })

    _add_efficiency_rows(rows, records, reduction_by_mode, modes)
    _add_phase_count_rows(rows, phase_counts, modes)
    _add_is_rows(rows, n_removed_by_mode, reduction_by_mode, modes)
    return pd.DataFrame(rows)


def _empty_row(mode: str, metric: str) -> dict:
    return {
        "mode": mode, "metric": metric,
        "mean": float("nan"), "std": float("nan"),
        "ci_95_low": float("nan"), "ci_95_high": float("nan"),
        "n_folds": 0,
    }


def _add_efficiency_rows(rows, records, reduction_by_mode, modes):
    for mode in modes:
        f1_vals = records[mode].get("macro_f1", [])
        time_vals = records[mode].get("train_time_s", [])
        mean_f1 = float(np.mean(f1_vals)) if f1_vals else float("nan")
        mean_time_min = float(np.mean(time_vals)) / 60.0 if time_vals else 0.0
        eff = mean_f1 / mean_time_min if mean_time_min > 0 else float("nan")
        rows.append({**_empty_row(mode, "efficiency_score"), "mean": eff, "n_folds": len(f1_vals)})

        reduction = reduction_by_mode.get(mode, 0.0)
        data_eff = mean_f1 * (1.0 - reduction) if f1_vals else float("nan")
        rows.append({**_empty_row(mode, "data_efficiency"), "mean": data_eff, "n_folds": len(f1_vals)})


def _add_is_rows(rows, n_removed_by_mode, reduction_by_mode, modes):
    for mode in modes:
        removed_vals = n_removed_by_mode.get(mode, [])
        if removed_vals:
            arr = np.array(removed_vals, dtype=float)
            rows.append({
                "mode": mode, "metric": "n_removed",
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "ci_95_low": float("nan"), "ci_95_high": float("nan"),
                "n_folds": len(arr),
            })
        else:
            rows.append(_empty_row(mode, "n_removed"))

        reduction = reduction_by_mode.get(mode)
        if reduction is not None:
            rows.append({
                "mode": mode, "metric": "is_reduction",
                "mean": float(reduction),
                "std": 0.0,
                "ci_95_low": float("nan"), "ci_95_high": float("nan"),
                "n_folds": len(removed_vals) if removed_vals else 0,
            })
        else:
            rows.append(_empty_row(mode, "is_reduction"))


def _add_phase_count_rows(rows, phase_counts, modes):
    for mode in modes:
        counts = phase_counts.get(mode, [])
        if not counts:
            rows.append(_empty_row(mode, "n_phases"))
            continue
        arr = np.array(counts, dtype=float)
        rows.append({
            "mode": mode, "metric": "n_phases",
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "ci_95_low": float("nan"), "ci_95_high": float("nan"),
            "n_folds": len(arr),
        })


def print_summary(summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY  (95% CI via t-Student)")
    print("=" * 70)

    def _pivot(metric: str, cols=None):
        if cols is None:
            cols = ["mean", "ci_95_low", "ci_95_high"]
        sub = summary_df[summary_df["metric"] == metric]
        if sub.empty:
            print(f"{metric}: (no data)")
            return
        available = [c for c in cols if c in sub.columns]
        print(sub.set_index("mode")[available].to_string())

    print("Macro-F1 (best phase):")
    _pivot("macro_f1", ["mean", "ci_95_low", "ci_95_high", "n_folds"])
    print("\nEfficiency Score (macro-F1 / train-min):")
    _pivot("efficiency_score", ["mean"])
    print("\nData Efficiency (macro-F1 * data fraction kept):")
    _pivot("data_efficiency", ["mean"])
    print("=" * 70)
