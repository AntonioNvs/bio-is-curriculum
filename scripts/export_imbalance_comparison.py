"""Export imbalance-method comparisons to a single .xlsx workbook.

This script scans `results/` for experiment folders in the format:
    <dataset>-<n>cv-<imbalance_method>-<timestamp>

By default it picks the latest available run for each
(dataset, imbalance_method) pair and creates one worksheet per dataset.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_DATASETS = ("reuters90", "ohsumed")
DEFAULT_METHODS = (
    "inverse_freq_ce",
    "effective_num_cb",
    "distribution_balanced",
    "minority_eda",
)
DEFAULT_METRICS = (
    "macro_f1",
    "micro_f1",
    "f1_weighted",
    "accuracy",
    "hard_slice_macro_f1",
    "train_time_s",
    "efficiency_score",
    "data_efficiency",
    "n_phases",
)

_EXP_RE = re.compile(
    r"^(?P<dataset>.+)-(?P<n_splits>\d+)cv-(?P<method>[a-z0-9_]+)-(?P<ts>\d{8}-\d{6})$"
)


@dataclass(frozen=True)
class RunInfo:
    dataset: str
    method: str
    timestamp: str
    path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an .xlsx comparison of imbalance methods for selected datasets."
        )
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument(
        "--select",
        choices=("latest", "all"),
        default="latest",
        help="Use only the latest run per dataset/method, or all discovered runs.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _discover_runs(results_dir: Path, datasets: list[str], methods: list[str]) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for path in sorted(results_dir.iterdir()):
        if not path.is_dir():
            continue
        match = _EXP_RE.match(path.name)
        if not match:
            continue
        dataset = match.group("dataset")
        method = match.group("method")
        ts = match.group("ts")
        if dataset not in datasets or method not in methods:
            continue
        if not (path / "summary.csv").exists():
            continue
        runs.append(RunInfo(dataset=dataset, method=method, timestamp=ts, path=path))
    return runs


def _select_runs(runs: list[RunInfo], select_mode: str) -> list[RunInfo]:
    if select_mode == "all":
        return runs

    latest: dict[tuple[str, str], RunInfo] = {}
    for run in runs:
        key = (run.dataset, run.method)
        prev = latest.get(key)
        if prev is None or run.timestamp > prev.timestamp:
            latest[key] = run
    return sorted(latest.values(), key=lambda r: (r.dataset, r.method, r.timestamp))


def _collect_rows(runs: list[RunInfo], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in runs:
        summary_path = run.path / "summary.csv"
        df = pd.read_csv(summary_path)
        df = df[df["metric"].isin(metrics)]
        for _, row in df.iterrows():
            rows.append(
                {
                    "dataset": run.dataset,
                    "imbalance_method": run.method,
                    "mode": row["mode"],
                    "metric": row["metric"],
                    "mean": row["mean"],
                    "ci_95_low": row.get("ci_95_low"),
                    "ci_95_high": row.get("ci_95_high"),
                    "n_folds": row.get("n_folds"),
                    "experiment_id": run.path.name,
                    "run_timestamp": run.timestamp,
                }
            )
    return pd.DataFrame(rows)


def _build_dataset_sheet(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if df.empty:
        return df

    merged: pd.DataFrame | None = None
    for metric in metrics:
        metric_df = df[df["metric"] == metric][
            [
                "imbalance_method",
                "mode",
                "mean",
                "ci_95_low",
                "ci_95_high",
                "n_folds",
                "experiment_id",
            ]
        ].rename(
            columns={
                "mean": f"{metric}_mean",
                "ci_95_low": f"{metric}_ci_95_low",
                "ci_95_high": f"{metric}_ci_95_high",
                "n_folds": f"{metric}_n_folds",
                "experiment_id": f"{metric}_experiment_id",
            }
        )
        if merged is None:
            merged = metric_df
        else:
            merged = merged.merge(metric_df, on=["imbalance_method", "mode"], how="outer")

    assert merged is not None
    merged = merged.sort_values(by=["imbalance_method", "mode"]).reset_index(drop=True)
    return merged


def _default_output_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"imbalance-compare-reuters90-ohsumed-{ts}.xlsx")


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results-dir not found: {results_dir}")

    runs = _discover_runs(results_dir, args.datasets, args.methods)
    if not runs:
        raise ValueError("No matching experiment folders with summary.csv were found.")

    selected_runs = _select_runs(runs, args.select)
    result_df = _collect_rows(selected_runs, args.metrics)
    if result_df.empty:
        raise ValueError("No rows found for requested metrics.")

    output_path = Path(args.output) if args.output else _default_output_path()
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        raw_sheet = result_df.sort_values(
            by=["dataset", "imbalance_method", "mode", "metric", "run_timestamp"]
        ).reset_index(drop=True)
        raw_sheet.to_excel(writer, sheet_name="raw_rows", index=False)

        for dataset in args.datasets:
            dataset_df = result_df[result_df["dataset"] == dataset].copy()
            if dataset_df.empty:
                continue
            wide = _build_dataset_sheet(dataset_df, args.metrics)
            wide.to_excel(writer, sheet_name=dataset[:31], index=False)

    print(f"Saved workbook: {output_path.resolve()}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Runs used: {len(selected_runs)}")


if __name__ == "__main__":
    main()
