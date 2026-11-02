"""Paired bootstrap equivalence test (TOST-style) between Raw and IS+CL.

Follows the advisor's design: for each dataset, compare the *per-instance*
predictions of Raw and IS+CL on the same test instances. Pool the test
instances across folds, bootstrap-resample instances with replacement,
recompute macro-F1 for both methods on each resample, and take
``Delta = MacroF1(IS+CL) - MacroF1(Raw)``. Equivalence is supported when the
90% confidence interval of ``Delta`` lies entirely within ``[-delta, +delta]``.

Inputs are the per-fold ``predictions_test.csv`` files (columns ``idx, y_true,
y_pred, pred_entropy``) written by ``RunRecorder.save_predictions``. Those files
must be regenerated (re-run the experiments) if absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bio_is_curriculum.results.aggregator import collect_per_fold_map, collect_per_fold_values
from bio_is_curriculum.results.discovery import identify_latest
from bio_is_curriculum.results.paper_tables import FACTORIAL_DATASETS, PAPER_TABLES

DEFAULT_DELTA = 0.01
DEFAULT_CI = 0.90
DEFAULT_N_BOOTSTRAP = 10_000
DEFAULT_SEED = 42

_FACTORIAL = next(t for t in PAPER_TABLES if t.name == "factorial")


def _remap(y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Map class ids to ``0..K-1`` (unknown -> -1)."""
    index = {int(c): i for i, c in enumerate(labels)}
    return np.array([index.get(int(v), -1) for v in y], dtype=np.int64)


def _macro_f1_mapped(t: np.ndarray, p: np.ndarray, k: int) -> float:
    valid = (t >= 0) & (p >= 0)
    t = t[valid]
    p = p[valid]
    cm = np.bincount(t * k + p, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(prec), where=denom > 0)
    return float(np.mean(f1))


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Read a ``predictions_test.csv`` into a DataFrame (idx, y_true, y_pred)."""
    df = pd.read_csv(path)
    for col in ("idx", "y_true", "y_pred"):
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r} in {path}")
    return df.sort_values("idx")[["idx", "y_true", "y_pred"]].reset_index(drop=True)


def bootstrap_equivalence(
    raw_preds: list[tuple[np.ndarray, np.ndarray]],
    iscl_preds: list[tuple[np.ndarray, np.ndarray]],
    *,
    delta: float = DEFAULT_DELTA,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run the paired bootstrap on aligned ``(y_true, y_pred)`` fold arrays.

    ``raw_preds`` / ``iscl_preds`` are per-fold ``(y_true, y_pred)`` tuples;
    fold order and test instances must align between the two methods.
    """
    n_folds = len(raw_preds)
    if n_folds != len(iscl_preds):
        raise ValueError("Raw and IS+CL must have the same number of folds.")

    y_true_pool: list[np.ndarray] = []
    raw_pool: list[np.ndarray] = []
    iscl_pool: list[np.ndarray] = []
    for (yt_r, yr), (yt_i, yi) in zip(raw_preds, iscl_preds):
        if not np.array_equal(yt_r, yt_i):
            raise ValueError("Fold y_true mismatch between Raw and IS+CL.")
        y_true_pool.append(np.asarray(yt_r, dtype=np.int64))
        raw_pool.append(np.asarray(yr, dtype=np.int64))
        iscl_pool.append(np.asarray(yi, dtype=np.int64))

    y_true = np.concatenate(y_true_pool)
    raw = np.concatenate(raw_pool)
    iscl = np.concatenate(iscl_pool)
    labels = np.unique(y_true)
    k = len(labels)
    n = len(y_true)

    t = _remap(y_true, labels)
    raw_m = _remap(raw, labels)
    iscl_m = _remap(iscl, labels)

    f1_raw_full = _macro_f1_mapped(t, raw_m, k)
    f1_iscl_full = _macro_f1_mapped(t, iscl_m, k)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        f1_raw = _macro_f1_mapped(t[idx], raw_m[idx], k)
        f1_iscl = _macro_f1_mapped(t[idx], iscl_m[idx], k)
        diffs[b] = f1_iscl - f1_raw

    low_q = ((1.0 - ci) / 2.0) * 100.0
    high_q = ((1.0 + ci) / 2.0) * 100.0
    ci_low, ci_high = float(np.percentile(diffs, low_q)), float(np.percentile(diffs, high_q))
    equivalent = bool(ci_low >= -delta and ci_high <= delta)

    return {
        "n_folds": n_folds,
        "n_instances": n,
        "n_classes": k,
        "macro_f1_raw": f1_raw_full,
        "macro_f1_iscl": f1_iscl_full,
        "delta_hat": f1_iscl_full - f1_raw_full,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": ci,
        "margin": delta,
        "equivalent": equivalent,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_equivalence_folds(
    raw_f1: np.ndarray,
    iscl_f1: np.ndarray,
    *,
    delta: float = DEFAULT_DELTA,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Fold-level paired bootstrap equivalence (degraded fallback).

    Used when per-instance ``predictions_test.csv`` files are absent. Resamples
    the per-fold ``macro_f1`` values (one per fold, paired) instead of test
    instances. With only ~5 folds this is coarse: the CI reflects fold-to-fold
    variability, not instance-level variability, so it is *not* the advisor's
    procedure and should be reported as such.
    """
    n_folds = len(raw_f1)
    if n_folds != len(iscl_f1):
        raise ValueError("Raw and IS+CL must have the same number of folds.")
    if n_folds < 2:
        raise ValueError("Fold-level bootstrap requires at least 2 folds.")
    raw = np.asarray(raw_f1, dtype=float)
    iscl = np.asarray(iscl_f1, dtype=float)

    delta_hat = float(np.mean(iscl) - np.mean(raw))

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_folds, n_folds)
        diffs[b] = np.mean(iscl[idx]) - np.mean(raw[idx])

    low_q = ((1.0 - ci) / 2.0) * 100.0
    high_q = ((1.0 + ci) / 2.0) * 100.0
    ci_low, ci_high = float(np.percentile(diffs, low_q)), float(np.percentile(diffs, high_q))
    equivalent = bool(ci_low >= -delta and ci_high <= delta)

    return {
        "n_folds": n_folds,
        "n_instances": n_folds,
        "n_classes": 0,
        "macro_f1_raw": float(np.mean(raw)),
        "macro_f1_iscl": float(np.mean(iscl)),
        "delta_hat": delta_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": ci,
        "margin": delta,
        "equivalent": equivalent,
        "n_bootstrap": n_bootstrap,
    }


def resolve_pair_runs(
    results_dir: str | Path,
    dataset: str,
    n_splits: int = 5,
) -> dict[str, tuple[str, str]] | None:
    """Resolve the factorial ``Raw`` and ``IS+CL`` run dirs for one dataset.

    Returns ``{"Raw": (path, mode), "IS+CL": (path, mode)}`` or ``None`` if
    either method is unresolved.
    """
    mapping = identify_latest(results_dir, tables=[_FACTORIAL], n_splits=n_splits)
    sub = mapping[mapping["dataset"] == dataset]
    out: dict[str, tuple[str, str]] = {}
    for method in ("Raw", "IS+CL"):
        row = sub[sub["method"] == method]
        if row.empty or not row.iloc[0]["path"]:
            return None
        out[method] = (str(row.iloc[0]["path"]), str(row.iloc[0]["mode"]))
    return out


def _load_fold_pair(
    raw_dir: str, raw_mode: str, iscl_dir: str, iscl_mode: str, fold: int
) -> tuple[np.ndarray, np.ndarray] | None:
    raw_path = Path(raw_dir) / f"{raw_mode}_fold{fold}" / "predictions_test.csv"
    iscl_path = Path(iscl_dir) / f"{iscl_mode}_fold{fold}" / "predictions_test.csv"
    if not raw_path.exists() or not iscl_path.exists():
        return None
    raw_df = load_predictions(raw_path)
    iscl_df = load_predictions(iscl_path)
    if len(raw_df) != len(iscl_df) or not (raw_df["idx"] == iscl_df["idx"]).all():
        raise ValueError(f"Prediction idx misalignment in fold {fold}.")
    return (raw_df["y_true"].to_numpy(), raw_df["y_pred"].to_numpy()), (
        iscl_df["y_true"].to_numpy(),
        iscl_df["y_pred"].to_numpy(),
    )


def run(
    results_dir: str | Path = "results",
    datasets: list[str] | None = None,
    *,
    delta: float = DEFAULT_DELTA,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Run the Raw vs IS+CL bootstrap equivalence per dataset.

    Uses per-instance ``predictions_test.csv`` when available (the advisor's
    procedure). When those files are absent, falls back to a *fold-level*
    bootstrap over the surviving per-fold ``macro_f1`` values, flagged
    ``bootstrap_level == "fold"`` (degraded, low power with ~5 folds).
    """
    datasets = datasets or FACTORIAL_DATASETS
    rows: list[dict] = []
    for dataset in datasets:
        pair = resolve_pair_runs(results_dir, dataset, n_splits=n_splits)
        if pair is None:
            rows.append(
                {
                    "dataset": dataset,
                    "status": "unresolved",
                    "bootstrap_level": "none",
                    "dataset_size": float("nan"),
                    "n_folds": 0,
                    "n_instances": 0,
                    "n_classes": 0,
                    "macro_f1_raw": float("nan"),
                    "macro_f1_iscl": float("nan"),
                    "delta_hat": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "ci_level": ci,
                    "margin": delta,
                    "equivalent": False,
                    "n_bootstrap": 0,
                    "train_time_raw_s": float("nan"),
                    "train_time_iscl_s": float("nan"),
                }
            )
            continue

        raw_dir, raw_mode = pair["Raw"]
        iscl_dir, iscl_mode = pair["IS+CL"]
        folds_range = list(range(n_splits))

        raw_sizes = collect_per_fold_values(raw_dir, raw_mode, folds_range, "n_train_instances")
        if raw_sizes:
            dataset_size = round(float(np.mean(raw_sizes)))
        else:
            # Older-schema runs (e.g. medline Raw) lack n_train_instances; fall back
            # to the IS+CL run's final-phase training size as a proxy for dataset size.
            iscl_sizes = collect_per_fold_values(iscl_dir, iscl_mode, folds_range, "n_train_instances")
            dataset_size = round(float(np.mean(iscl_sizes))) if iscl_sizes else float("nan")

        raw_preds, iscl_preds = [], []
        for fold in folds_range:
            loaded = _load_fold_pair(raw_dir, raw_mode, iscl_dir, iscl_mode, fold)
            if loaded is None:
                continue
            raw_preds.append(loaded[0])
            iscl_preds.append(loaded[1])

        raw_times = collect_per_fold_values(raw_dir, raw_mode, folds_range, "train_time_s")
        iscl_times = collect_per_fold_values(iscl_dir, iscl_mode, folds_range, "train_time_s")
        time_row = {
            "train_time_raw_s": float(np.mean(raw_times)) if raw_times else float("nan"),
            "train_time_iscl_s": float(np.mean(iscl_times)) if iscl_times else float("nan"),
        }

        if raw_preds:
            # Advisor's procedure: per-instance paired bootstrap.
            result = bootstrap_equivalence(
                raw_preds, iscl_preds, delta=delta, n_bootstrap=n_bootstrap, ci=ci, seed=seed
            )
            rows.append(
                {
                    "dataset": dataset,
                    "status": "ok",
                    "bootstrap_level": "instance",
                    "dataset_size": dataset_size,
                    **result,
                    **time_row,
                }
            )
            continue

        # Fallback: fold-level bootstrap over per-fold macro-F1 (degraded).
        raw_map = collect_per_fold_map(raw_dir, raw_mode, folds_range, "macro_f1")
        iscl_map = collect_per_fold_map(iscl_dir, iscl_mode, folds_range, "macro_f1")
        shared_folds = sorted(set(raw_map) & set(iscl_map))
        if len(shared_folds) >= 2:
            raw_f1 = np.array([raw_map[f] for f in shared_folds], dtype=float)
            iscl_f1 = np.array([iscl_map[f] for f in shared_folds], dtype=float)
            result = bootstrap_equivalence_folds(
                raw_f1, iscl_f1, delta=delta, n_bootstrap=n_bootstrap, ci=ci, seed=seed
            )
            rows.append(
                {
                    "dataset": dataset,
                    "status": "ok",
                    "bootstrap_level": "fold",
                    "dataset_size": dataset_size,
                    **result,
                    **time_row,
                }
            )
        else:
            rows.append(
                {
                    "dataset": dataset,
                    "status": "missing_data",
                    "bootstrap_level": "none",
                    "dataset_size": dataset_size,
                    "n_folds": len(shared_folds),
                    "n_instances": 0,
                    "n_classes": 0,
                    "macro_f1_raw": float("nan"),
                    "macro_f1_iscl": float("nan"),
                    "delta_hat": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "ci_level": ci,
                    "margin": delta,
                    "equivalent": False,
                    "n_bootstrap": 0,
                    **time_row,
                }
            )

    cols = [
        "dataset",
        "status",
        "bootstrap_level",
        "dataset_size",
        "n_folds",
        "n_instances",
        "n_classes",
        "macro_f1_raw",
        "macro_f1_iscl",
        "delta_hat",
        "ci_low",
        "ci_high",
        "ci_level",
        "margin",
        "equivalent",
        "n_bootstrap",
        "train_time_raw_s",
        "train_time_iscl_s",
    ]
    return pd.DataFrame(rows, columns=cols)


def _print_report(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(
        df[
            [
                "dataset",
                "status",
                "bootstrap_level",
                "dataset_size",
                "n_instances",
                "macro_f1_raw",
                "macro_f1_iscl",
                "delta_hat",
                "ci_low",
                "ci_high",
                "margin",
                "equivalent",
                "train_time_raw_s",
                "train_time_iscl_s",
            ]
        ].to_string(index=False)
    )
    if (df["bootstrap_level"] == "fold").any():
        print(
            "\nWARNING: one or more rows used bootstrap_level='fold' (per-fold macro-F1"
            "\nresampled over ~5 folds) because per-instance predictions_test.csv files are"
            "\nmissing. This is NOT the advisor's per-instance procedure; CIs are coarse and"
            "\n'equivalent=False' here mostly means 'inconclusive', not 'provably worse'."
        )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Paired bootstrap equivalence (TOST) between Raw and IS+CL on "
            "per-instance test predictions."
        )
    )
    parser.add_argument("--results-dir", default="results", help="Base results directory.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Limit datasets.")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="Equivalence margin.")
    parser.add_argument("--ci", type=float, default=DEFAULT_CI, help="Confidence level.")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP, help="Bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed.")
    parser.add_argument("--n-splits", type=int, default=5, help="Expected CV folds.")
    parser.add_argument("--output", default=None, help="Optional CSV path for results.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    df = run(
        results_dir=args.results_dir,
        datasets=args.datasets,
        delta=args.delta,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        seed=args.seed,
        n_splits=args.n_splits,
    )
    if args.output:
        df.to_csv(args.output, index=False, float_format="%.6f")
        print(f"Saved equivalence results to: {args.output}")
    _print_report(df)


if __name__ == "__main__":
    main()
