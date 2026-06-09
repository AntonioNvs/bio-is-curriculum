"""Aggregate experiment summaries into a single Excel file.

Example:
    python summary.py \
        --metric macro_f1 \
        webkb-10cv-20260605-011430-0815f0 \
        webkb-10cv-20260607-191540-731d44 \
        webkb-10cv-20260608-024640-c22b92
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
    "mpqa-10cv-20260608-131847-b85a8b"
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read multiple results/<experiment_id>/summary.csv files and "
            "export one consolidated .xlsx with dataset/method and confidence interval."
        )
    )
    parser.add_argument(
        "folders",
        nargs="*",
        help=(
            "Experiment folder names inside results/ (or absolute paths). "
            "If omitted, uses the default FOLDERS list from this file."
        ),
    )
    parser.add_argument(
        "--metric",
        required=True,
        help=(
            "Metric to extract. For regular metrics (example: macro_f1), reads summary.csv. "
            "For total_time, reads *_fold*/timings.csv and uses total_run_time_s."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Base directory for result folders (default: results).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output .xlsx path. If omitted, a unique filename is generated "
            "(summary-<metric>-YYYYmmdd-HHMMSS.xlsx)."
        ),
    )
    return parser.parse_args()


def _resolve_folder(folder_arg: str, results_dir: Path) -> Path:
    return results_dir / folder_arg


def _dataset_from_experiment_id(experiment_id: str) -> str:
    # Typical format: <dataset>-<n>cv-<timestamp>-<hash>
    match = re.match(r"^(.*)-\d+cv-", experiment_id)
    if match:
        return match.group(1)
    return experiment_id


def _collect_mode_metadata(experiment_dir: Path) -> tuple[str | None, dict[str, str | None]]:
    config_paths = sorted(experiment_dir.glob("*_fold*/config.json"))
    if not config_paths:
        return None, {}

    dataset: str | None = None
    mode_to_method: dict[str, str | None] = {}

    for config_path in config_paths:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        dataset = dataset or cfg.get("dataset")
        mode = cfg.get("mode")
        curriculum_method = cfg.get("curriculum_method")
        if mode and "cl" in mode and not curriculum_method:
            curriculum_method = DEFAULT_CURRICULUM_METHOD
        if mode and mode not in mode_to_method:
            mode_to_method[mode] = curriculum_method

    return dataset, mode_to_method


def _format_method_label(mode: str, curriculum_method: str | None) -> str:
    if "cl" in mode:
        label = curriculum_method or DEFAULT_CURRICULUM_METHOD
        return f"{mode} ({label})"
    return mode


def _extract_rows_for_metric(
    experiment_dir: Path,
    metric: str,
) -> list[dict[str, object]]:
    summary_path = experiment_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    if "metric" not in summary_df.columns:
        raise ValueError(f"Invalid summary.csv (missing 'metric' column): {summary_path}")

    metric_df = summary_df[summary_df["metric"] == metric].copy()
    if metric_df.empty:
        raise ValueError(f"Metric '{metric}' not found in: {summary_path}")

    dataset, mode_to_method = _collect_mode_metadata(experiment_dir)
    dataset = dataset or _dataset_from_experiment_id(experiment_dir.name)

    required_cols = {"mode", "mean", "ci_95_low", "ci_95_high"}
    missing_cols = required_cols.difference(metric_df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"Missing required columns ({missing}) in: {summary_path}")

    rows: list[dict[str, object]] = []
    for _, row in metric_df.iterrows():
        mode = str(row["mode"])
        curriculum_method = mode_to_method.get(mode)
        rows.append(
            {
                "dataset": dataset,
                "method": _format_method_label(mode, curriculum_method),
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


def _extract_rows_for_total_time(experiment_dir: Path) -> list[dict[str, object]]:
    dataset, mode_to_method = _collect_mode_metadata(experiment_dir)
    dataset = dataset or _dataset_from_experiment_id(experiment_dir.name)

    mode_values: dict[str, list[float]] = {}
    skipped_files: list[str] = []
    timing_paths = sorted(experiment_dir.glob("*_fold*/timings.csv"))
    if not timing_paths:
        raise FileNotFoundError(f"No timings.csv files found in: {experiment_dir}")

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
        curriculum_method = mode_to_method.get(mode)
        rows.append(
            {
                "dataset": dataset,
                "method": _format_method_label(mode, curriculum_method),
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
            "'total_run_time_s'/'total_time'."
        )
    return rows


def _default_output_path(metric: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"summary-{metric}-{ts}.xlsx")


def _experiment_sort_key(experiment_id: str) -> str:
    match = re.search(r"-(\d{8}-\d{6})-[^-]+$", experiment_id)
    return match.group(1) if match else experiment_id


def _deduplicate_rows(output_df: pd.DataFrame) -> pd.DataFrame:
    # Remove exact duplicates first.
    dedup = output_df.drop_duplicates().copy()
    # If the same dataset+method+metric appears across multiple experiment folders,
    # keep the latest run by timestamp embedded in experiment_id.
    dedup["_exp_sort"] = dedup["experiment_id"].astype(str).map(_experiment_sort_key)
    dedup = dedup.sort_values(by=["dataset", "method", "metric", "_exp_sort"])
    dedup = dedup.drop_duplicates(subset=["dataset", "method", "metric"], keep="last")
    dedup = dedup.drop(columns=["_exp_sort"])
    return dedup.reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)

    folders = args.folders if args.folders else FOLDERS
    if not folders:
        raise ValueError("No folders provided. Pass folders or define FOLDERS.")

    all_rows: list[dict[str, object]] = []
    for folder in folders:
        experiment_dir = _resolve_folder(folder, results_dir)
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment folder not found: {experiment_dir}")
        if args.metric == "total_time":
            all_rows.extend(_extract_rows_for_total_time(experiment_dir))
        else:
            all_rows.extend(_extract_rows_for_metric(experiment_dir, args.metric))

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


if __name__ == "__main__":
    main()
