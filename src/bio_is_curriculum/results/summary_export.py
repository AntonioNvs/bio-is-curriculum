"""Export experiment summaries from manifest JSON to Excel/CSV."""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bio_is_curriculum.results.manifest import (
    SUMMARY_CSV_NAME,
    SUMMARY_XLSX_NAME,
    load_manifest,
    resolve_manifest_path,
)

DEFAULT_CURRICULUM_METHOD = "biois_discrete"

_MODE_BASE_ORDER = {"raw": 0, "is": 1, "b1": 2, "cl": 3, "is_cl": 4}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an experiment manifest JSON and export consolidated .xlsx and .csv "
            "summaries. Legacy folder discovery is deprecated."
        )
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        default=None,
        help=(
            "Path to results/experiments/<event>_<timestamp>/ "
            "(folder) or .../manifest.json"
        ),
    )
    parser.add_argument(
        "folders",
        nargs="*",
        help="(Deprecated) Experiment folder names inside results/.",
    )
    parser.add_argument("--metric", default=None, help="(Deprecated legacy mode) Single metric.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Override manifest metrics.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="(Deprecated) Force compare_by_dataset layout.",
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="(Deprecated) Discover folders by timestamp substring.",
    )
    parser.add_argument("--datasets", nargs="*", default=None, help="Limit exported datasets.")
    parser.add_argument("--results-dir", default="results", help="Base results directory.")
    parser.add_argument("--output", default=None, help="Override output .xlsx path.")
    return parser.parse_args(argv)


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
            f"'total_run_time_s'/'total_time' in {experiment_dir.name}."
        )
    return rows


def extract_metric_rows(
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
    """Find experiment folders from a legacy batch run."""
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


def _experiment_sort_key(experiment_id: str) -> str:
    match = re.search(r"-(\d{8}-\d{6})", experiment_id)
    return match.group(1) if match else experiment_id


def deduplicate_rows(output_df: pd.DataFrame) -> pd.DataFrame:
    dedup = output_df.drop_duplicates().copy()
    dedup["_exp_sort"] = dedup["experiment_id"].astype(str).map(_experiment_sort_key)
    dedup = dedup.sort_values(by=["dataset", "method", "metric", "_exp_sort"])
    dedup = dedup.drop_duplicates(subset=["dataset", "method", "metric"], keep="last")
    dedup = dedup.drop(columns=["_exp_sort"])
    return dedup.reset_index(drop=True)


def pivot_compare_sheet(df: pd.DataFrame) -> pd.DataFrame:
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


def collect_rows_from_dirs(
    experiment_dirs: list[Path],
    metrics: list[str],
    *,
    strict: bool = False,
) -> pd.DataFrame:
    all_rows: list[dict[str, object]] = []
    for experiment_dir in experiment_dirs:
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment folder not found: {experiment_dir}")
        for metric in metrics:
            rows = extract_metric_rows(experiment_dir, metric, strict=strict)
            all_rows.extend(rows)
    if not all_rows:
        raise ValueError("No metrics collected. Check folders and metric names.")
    return deduplicate_rows(pd.DataFrame(all_rows))


def _manifest_experiment_dirs(manifest: dict[str, object]) -> list[Path]:
    runs = manifest.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("Manifest has no runs to export.")
    dirs: list[Path] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        path = run.get("path")
        if not path:
            continue
        dirs.append(Path(str(path)))
    if not dirs:
        raise ValueError("Manifest runs do not contain valid paths.")
    return dirs


def export_from_manifest(
    manifest_input: str | Path,
    *,
    metrics: list[str] | None = None,
    datasets: list[str] | None = None,
    output_xlsx: Path | None = None,
) -> tuple[Path, Path]:
    manifest_json = resolve_manifest_path(manifest_input)
    manifest = load_manifest(manifest_json)
    summary_cfg = manifest.get("summary") or {}
    if not isinstance(summary_cfg, dict):
        raise ValueError(f"Invalid summary section in manifest: {manifest_json}")

    layout = str(summary_cfg.get("layout", "long_table"))
    if metrics is None:
        raw_metrics = summary_cfg.get("metrics")
        metrics = list(raw_metrics) if raw_metrics else ["macro_f1"]
    manifest_datasets = summary_cfg.get("datasets")
    if datasets is None and isinstance(manifest_datasets, list):
        datasets = [str(d) for d in manifest_datasets]

    experiment_dirs = _manifest_experiment_dirs(manifest)
    long_df = collect_rows_from_dirs(experiment_dirs, metrics, strict=False)

    if datasets:
        long_df = long_df[long_df["dataset"].isin(datasets)].copy()
        if long_df.empty:
            raise ValueError(f"No rows for datasets: {datasets}")

    out_dir = manifest_json.parent
    csv_path = out_dir / SUMMARY_CSV_NAME
    xlsx_path = output_xlsx or (out_dir / SUMMARY_XLSX_NAME)

    long_df.sort_values(by=["dataset", "method", "metric"]).to_csv(
        csv_path, index=False, float_format="%.6f"
    )

    if layout == "compare_by_dataset":
        sheet_datasets = datasets or sorted(long_df["dataset"].dropna().unique().tolist())
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for dataset in sheet_datasets:
                sheet_df = long_df[long_df["dataset"] == dataset].copy()
                if sheet_df.empty:
                    print(f"Warning: no rows for dataset '{dataset}', skipping sheet.")
                    continue
                wide = pivot_compare_sheet(sheet_df)
                wide.to_excel(writer, sheet_name=dataset[:31], index=False)
    else:
        long_df.to_excel(xlsx_path, index=False)

    return xlsx_path, csv_path


def run_legacy(args: argparse.Namespace) -> None:
    warnings.warn(
        "Legacy folder/metric export is deprecated; pass a manifest JSON path instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not args.metric:
        raise ValueError("--metric is required for legacy export.")

    results_dir = Path(args.results_dir)
    if args.folders:
        experiment_dirs = [_resolve_folder(folder, results_dir) for folder in args.folders]
    elif args.run_prefix:
        experiment_dirs = discover_run_folders(results_dir, args.run_prefix, args.datasets)
        if not experiment_dirs:
            raise FileNotFoundError(
                f"No experiment folders matching run-prefix={args.run_prefix!r} under {results_dir}"
            )
    else:
        raise ValueError("Provide a manifest path or legacy folder arguments.")

    long_df = collect_rows_from_dirs(experiment_dirs, [args.metric], strict=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = Path(args.output) if args.output else Path(f"summary-{args.metric}-{ts}.xlsx")
    long_df.to_excel(output_path, index=False)
    print(f"Saved consolidated summary to: {output_path.resolve()}")
    print(f"Rows written: {len(long_df)}")


def run_legacy_compare(args: argparse.Namespace) -> None:
    warnings.warn(
        "Legacy --compare/--run-prefix export is deprecated; pass a manifest JSON path instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    results_dir = Path(args.results_dir)
    if args.folders:
        experiment_dirs = [_resolve_folder(folder, results_dir) for folder in args.folders]
    elif args.run_prefix:
        experiment_dirs = discover_run_folders(results_dir, args.run_prefix, args.datasets)
        if not experiment_dirs:
            raise FileNotFoundError(
                f"No experiment folders matching run-prefix={args.run_prefix!r} under {results_dir}"
            )
    else:
        raise ValueError("Provide a manifest path or legacy folder arguments.")

    metrics = args.metrics if args.metrics else [
        "macro_f1",
        "micro_f1",
        "f1_weighted",
        "accuracy",
        "hard_slice_macro_f1",
        "train_time_s",
        "total_time",
    ]
    long_df = collect_rows_from_dirs(experiment_dirs, metrics, strict=False)
    datasets = args.datasets or sorted(long_df["dataset"].dropna().unique().tolist())
    output_path = Path(args.output) if args.output else Path(
        f"summary-compare-{datetime.now():%Y%m%d-%H%M%S}.xlsx"
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for dataset in datasets:
            sheet_df = long_df[long_df["dataset"] == dataset].copy()
            if sheet_df.empty:
                continue
            wide = pivot_compare_sheet(sheet_df)
            wide.to_excel(writer, sheet_name=dataset[:31], index=False)
    print(f"Saved comparative summary to: {output_path.resolve()}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    manifest_arg = args.manifest
    if manifest_arg:
        path = Path(manifest_arg)
        if path.is_dir() or path.suffix == ".json":
            xlsx_path, csv_path = export_from_manifest(
                manifest_arg,
                metrics=args.metrics,
                datasets=args.datasets,
                output_xlsx=Path(args.output) if args.output else None,
            )
            print(f"Saved summary workbook to: {xlsx_path.resolve()}")
            print(f"Saved long-format CSV to: {csv_path.resolve()}")
            return
        args.folders = [manifest_arg, *list(args.folders)]

    if args.compare or args.run_prefix or args.folders or args.metric:
        if args.compare or args.run_prefix:
            run_legacy_compare(args)
        else:
            run_legacy(args)
        return

    raise SystemExit(
        "Usage: summary.py results/experiments/<event>_<timestamp>/ "
        "[--metrics ...] [--datasets ...]"
    )


if __name__ == "__main__":
    main()
