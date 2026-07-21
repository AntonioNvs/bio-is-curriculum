"""Aggregate experiment summaries into a single Excel file.

Examples:
    # Legacy: one metric, one sheet, explicit folders
    python summary.py --metric macro_f1 \\
        webkb-10cv-20260605-011430-0815f0 \\
        webkb-10cv-20260607-191540-731d44

    # Compare all methods from a multi-script batch (one sheet per dataset)
    python summary.py --compare --run-prefix 20260711-022935 \\
        --datasets webkb reuters90 \\
        --output summary-compare-20260711-022935.xlsx
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_CURRICULUM_METHOD = "biois_discrete"

DEFAULT_COMPARE_METRICS = [
    "macro_f1",
    "micro_f1",
    "f1_weighted",
    "accuracy",
    "hard_slice_macro_f1",
    "train_time_s",
    "total_time",
]

FOLDERS = [
    "webkb-10cv-20260605-011430-0815f0",
    "webkb-10cv-20260607-191540-731d44",
    "webkb-10cv-20260608-024640-c22b92",
    "yelp_reviews-10cv-20260605-214351-51bfad",
    "yelp_reviews-10cv-20260607-185414-02949a",
    "yelp_reviews-10cv-20260608-013035-ae35a2",
    "sst1-10cv-20260606-000740-994ca6",
    "sst1-10cv-20260608-021811-5f26d9",
    "reuters90-10cv-20260605-051521-3cbdc6",
    "reuters90-5cv-20260607-210619-22ded6",
    "reuters90-5cv-20260608-085030-684be2",
    "ohsumed-10cv-20260605-125242-e77409",
    "ohsumed-10cv-20260607-223026-0e699c",
    "ohsumed-10cv-20260608-141548-5173d8",
    "mpqa-10cv-20260605-120605-81b5f4",
    "mpqa-10cv-20260607-221351-5abce8",
    "mpqa-10cv-20260608-131847-b85a8b",
]

_MODE_BASE_ORDER = {"raw": 0, "is": 1, "b1": 2, "cl": 3, "is_cl": 4}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read results/<experiment_id>/summary.csv files and export a consolidated .xlsx. "
            "Use --compare for one sheet per dataset with all methods side by side."
        )
    )
    parser.add_argument(
        "folders",
        nargs="*",
        help=(
            "Experiment folder names inside results/ (or absolute paths). "
            "If omitted, uses --run-prefix discovery or the default FOLDERS list."
        ),
    )
    parser.add_argument(
        "--metric",
        default=None,
        help=(
            "Single metric (legacy mode). For regular metrics reads summary.csv; "
            "for total_time reads timings.csv."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help=f"Metrics for --compare (default: {DEFAULT_COMPARE_METRICS}).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Export one worksheet per dataset comparing all methods (wide format).",
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help=(
            "Discover experiment folders whose names contain this timestamp "
            "(e.g. 20260711-022935 from run_docker_full_cv_multi.sh)."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Limit --run-prefix discovery to these dataset names (e.g. webkb reuters90).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Base directory for result folders (default: results).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .xlsx path. If omitted, a timestamped filename is generated.",
    )
    return parser.parse_args()


def _resolve_folder(folder_arg: str, results_dir: Path) -> Path:
    path = Path(folder_arg)
    if path.is_absolute() or path.exists():
        return path
    return results_dir / folder_arg


def _dataset_from_experiment_id(experiment_id: str) -> str:
    match = re.match(r"^(.*)-\d+cv-", experiment_id)
    if match:
        return match.group(1)
    return experiment_id


def _collect_mode_metadata(
    experiment_dir: Path,
) -> tuple[str | None, dict[str, dict[str, str | None]]]:
    config_paths = sorted(experiment_dir.glob("*_fold*/config.json"))
    if not config_paths:
        return None, {}

    dataset: str | None = None
    mode_to_meta: dict[str, dict[str, str | None]] = {}

    for config_path in config_paths:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        dataset = dataset or cfg.get("dataset")
        mode = cfg.get("mode")
        if not mode:
            continue

        curriculum_method = cfg.get("curriculum_method")
        if "cl" in mode and not curriculum_method:
            curriculum_method = DEFAULT_CURRICULUM_METHOD

        if mode not in mode_to_meta:
            mode_to_meta[mode] = {
                "curriculum_method": curriculum_method,
                "scheme": cfg.get("curriculum_loss_scheme"),
            }

    return dataset, mode_to_meta


def _format_method_label(
    mode: str,
    meta: dict[str, str | None] | None,
) -> str:
    if "cl" not in mode:
        return mode

    curriculum_method = (meta or {}).get("curriculum_method") or DEFAULT_CURRICULUM_METHOD
    scheme = (meta or {}).get("scheme")
    label = curriculum_method
    if scheme:
        label = f"{curriculum_method}/{scheme}"
    return f"{mode} ({label})"


def _method_sort_key(label: str) -> tuple[int, str]:
    base = label.split(" (", 1)[0]
    return (_MODE_BASE_ORDER.get(base, 99), label)


def _extract_rows_for_metric(
    experiment_dir: Path,
    metric: str,
    *,
    strict: bool = True,
) -> list[dict[str, object]]:
    summary_path = experiment_dir / "summary.csv"
    if not summary_path.exists():
        if strict:
            raise FileNotFoundError(f"summary.csv not found: {summary_path}")
        return []

    summary_df = pd.read_csv(summary_path)
    if "metric" not in summary_df.columns:
        raise ValueError(f"Invalid summary.csv (missing 'metric' column): {summary_path}")

    metric_df = summary_df[summary_df["metric"] == metric].copy()
    if metric_df.empty:
        if strict:
            raise ValueError(f"Metric '{metric}' not found in: {summary_path}")
        return []

    dataset, mode_to_meta = _collect_mode_metadata(experiment_dir)
    dataset = dataset or _dataset_from_experiment_id(experiment_dir.name)

    required_cols = {"mode", "mean", "ci_95_low", "ci_95_high"}
    missing_cols = required_cols.difference(metric_df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"Missing required columns ({missing}) in: {summary_path}")

    rows: list[dict[str, object]] = []
    for _, row in metric_df.iterrows():
        mode = str(row["mode"])
        rows.append(
            {
                "dataset": dataset,
                "method": _format_method_label(mode, mode_to_meta.get(mode)),
                "mean": row["mean"],
                "ic_low": row["ci_95_low"],
                "ic_high": row["ci_95_high"],
                "metric": metric,
                "experiment_id": experiment_dir.name,
            }
        )

    return rows


def _compute_mean_ci(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        v = float(values[0])
        return v, float("nan"), float("nan")

    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    margin = t_crit * std / math.sqrt(n)
    return mean, mean - margin, mean + margin


def _extract_rows_for_total_time(
    experiment_dir: Path,
    *,
    strict: bool = True,
) -> list[dict[str, object]]:
    dataset, mode_to_meta = _collect_mode_metadata(experiment_dir)
    dataset = dataset or _dataset_from_experiment_id(experiment_dir.name)

    mode_values: dict[str, list[float]] = {}
    skipped_files: list[str] = []
    timing_paths = sorted(experiment_dir.glob("*_fold*/timings.csv"))
    if not timing_paths:
        if strict:
            raise FileNotFoundError(f"No timings.csv files found in: {experiment_dir}")
        return []

    for timing_path in timing_paths:
        mode = timing_path.parent.name.split("_fold")[0]
        timings_df = pd.read_csv(timing_path)
        if not {"name", "seconds"}.issubset(timings_df.columns):
            raise ValueError(f"Invalid timings.csv format in: {timing_path}")

        total_row = timings_df[timings_df["name"] == "total_run_time_s"]
        if total_row.empty:
            total_row = timings_df[timings_df["name"] == "total_time"]
        if total_row.empty:
            skipped_files.append(str(timing_path))
            continue

        seconds = float(total_row.iloc[0]["seconds"])
        mode_values.setdefault(mode, []).append(seconds)

    rows: list[dict[str, object]] = []
    for mode, values in sorted(mode_values.items()):
        mean, ci_low, ci_high = _compute_mean_ci(values)
        rows.append(
            {
                "dataset": dataset,
                "method": _format_method_label(mode, mode_to_meta.get(mode)),
                "mean": mean,
                "ic_low": ci_low,
                "ic_high": ci_high,
                "metric": "total_time",
                "experiment_id": experiment_dir.name,
            }
        )
    if skipped_files:
        print(
            f"Warning: skipped {len(skipped_files)} timings.csv file(s) without "
            "'total_run_time_s'/'total_time' in {experiment_dir.name}."
        )
    return rows


def _extract_metric_rows(
    experiment_dir: Path,
    metric: str,
    *,
    strict: bool,
) -> list[dict[str, object]]:
    if metric == "total_time":
        return _extract_rows_for_total_time(experiment_dir, strict=strict)
    return _extract_rows_for_metric(experiment_dir, metric, strict=strict)


def discover_run_folders(
    results_dir: Path,
    run_prefix: str,
    datasets: list[str] | None = None,
) -> list[Path]:
    """Find experiment folders from a run_docker_full_cv_multi batch."""
    candidates: list[Path] = []
    for path in sorted(results_dir.iterdir()):
        if not path.is_dir():
            continue
        if run_prefix not in path.name:
            continue
        if datasets and not any(path.name.startswith(f"{ds}-") for ds in datasets):
            continue
        if not path.glob("*_fold*/config.json") and not (path / "summary.csv").exists():
            continue
        candidates.append(path)
    return candidates


def _default_output_path(metric: str | None = None, *, compare: bool = False) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if compare:
        return Path(f"summary-compare-{ts}.xlsx")
    return Path(f"summary-{metric}-{ts}.xlsx")


def _experiment_sort_key(experiment_id: str) -> str:
    match = re.search(r"-(\d{8}-\d{6})", experiment_id)
    return match.group(1) if match else experiment_id


def _deduplicate_rows(output_df: pd.DataFrame) -> pd.DataFrame:
    dedup = output_df.drop_duplicates().copy()
    dedup["_exp_sort"] = dedup["experiment_id"].astype(str).map(_experiment_sort_key)
    dedup = dedup.sort_values(by=["dataset", "method", "metric", "_exp_sort"])
    dedup = dedup.drop_duplicates(subset=["dataset", "method", "metric"], keep="last")
    dedup = dedup.drop(columns=["_exp_sort"])
    return dedup.reset_index(drop=True)


def _pivot_compare_sheet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    merged: pd.DataFrame | None = None
    for metric in df["metric"].drop_duplicates():
        sub = df[df["metric"] == metric][
            ["method", "mean", "ic_low", "ic_high", "experiment_id"]
        ].rename(
            columns={
                "mean": f"{metric}_mean",
                "ic_low": f"{metric}_ci_low",
                "ic_high": f"{metric}_ci_high",
                "experiment_id": f"{metric}_experiment_id",
            }
        )
        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub, on="method", how="outer")

    assert merged is not None
    merged = merged.sort_values(
        "method",
        key=lambda col: col.map(_method_sort_key),
    ).reset_index(drop=True)
    return merged


def _resolve_experiment_dirs(args: argparse.Namespace, results_dir: Path) -> list[Path]:
    if args.folders:
        return [_resolve_folder(folder, results_dir) for folder in args.folders]

    if args.run_prefix:
        dirs = discover_run_folders(results_dir, args.run_prefix, args.datasets)
        if not dirs:
            raise FileNotFoundError(
                f"No experiment folders matching run-prefix={args.run_prefix!r} "
                f"under {results_dir}"
            )
        return dirs

    return [_resolve_folder(folder, results_dir) for folder in FOLDERS]


def run_compare(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    experiment_dirs = _resolve_experiment_dirs(args, results_dir)
    metrics = args.metrics if args.metrics else DEFAULT_COMPARE_METRICS

    all_rows: list[dict[str, object]] = []
    for experiment_dir in experiment_dirs:
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment folder not found: {experiment_dir}")
        for metric in metrics:
            rows = _extract_metric_rows(experiment_dir, metric, strict=False)
            all_rows.extend(rows)

    long_df = pd.DataFrame(all_rows)
    if long_df.empty:
        raise ValueError("No metrics collected. Check folders and metric names.")

    before = len(long_df)
    long_df = _deduplicate_rows(long_df)
    removed = before - len(long_df)

    datasets = args.datasets
    if not datasets:
        datasets = sorted(long_df["dataset"].dropna().unique().tolist())

    output_path = Path(args.output) if args.output else _default_output_path(compare=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for dataset in datasets:
            sheet_df = long_df[long_df["dataset"] == dataset].copy()
            if sheet_df.empty:
                print(f"Warning: no rows for dataset '{dataset}', skipping sheet.")
                continue
            wide = _pivot_compare_sheet(sheet_df)
            sheet_name = dataset[:31]
            wide.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved comparative summary to: {output_path.resolve()}")
    print(f"Experiment folders: {len(experiment_dirs)}")
    print(f"Datasets/sheets: {', '.join(datasets)}")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Methods (unique): {long_df['method'].nunique()}")
    if removed > 0:
        print(f"Duplicate rows removed: {removed}")


def run_legacy(args: argparse.Namespace) -> None:
    if not args.metric:
        raise ValueError("--metric is required unless --compare or --run-prefix is used.")

    results_dir = Path(args.results_dir)
    experiment_dirs = _resolve_experiment_dirs(args, results_dir)

    all_rows: list[dict[str, object]] = []
    for experiment_dir in experiment_dirs:
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment folder not found: {experiment_dir}")
        all_rows.extend(_extract_metric_rows(experiment_dir, args.metric, strict=True))

    output_df = pd.DataFrame(all_rows)
    before = len(output_df)
    output_df = _deduplicate_rows(output_df)
    removed = before - len(output_df)
    output_df = output_df.sort_values(by=["dataset", "method", "experiment_id"]).reset_index(drop=True)

    output_path = Path(args.output) if args.output else _default_output_path(args.metric)
    output_df.to_excel(output_path, index=False)

    print(f"Saved consolidated summary to: {output_path.resolve()}")
    print(f"Rows written: {len(output_df)}")
    if removed > 0:
        print(f"Duplicates removed: {removed}")


def main() -> None:
    args = _parse_args()
    if args.compare or args.run_prefix:
        run_compare(args)
    else:
        run_legacy(args)


if __name__ == "__main__":
    main()
