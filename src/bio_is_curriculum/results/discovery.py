"""Resolve the most recent result directory for each paper-table cell.

Result directories live under ``results/<dataset>-<ncv>cv-<timestamp>[-<variant>]/``
and contain one ``<mode>_fold<N>/`` subdirectory per fold with a ``config.json``
and a ``phase_metrics.csv``. This module scans those directories and, for every
``(dataset, mode, curriculum_method, extra config fields)`` requested by
``paper_tables``, returns the newest complete run.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from bio_is_curriculum.results.paper_tables import (
    PAPER_TABLES,
    REQUIRED_CONFIG,
    TableSpec,
)

_FOLD_RE = re.compile(r"^(?P<mode>.+)_fold(?P<fold>\d+)$")

# Subsampling / diagnostic runs to exclude from paper tables.
_EXCLUDED_TRAIN_FRACTION = 0.2


@dataclass
class RunRecord:
    """Indexed facts about a single result directory."""

    experiment_id: str
    path: Path
    dataset: str
    n_splits: int
    train_fraction: float | None
    curriculum_method: str
    timestamp: str
    modes: set[str] = field(default_factory=set)
    complete_modes: dict[str, int] = field(default_factory=dict)  # mode -> # folds
    _config: dict | None = None

    def config(self) -> dict:
        if self._config is None:
            with open(self._first_config_path(), encoding="utf-8") as f:
                self._config = json.load(f)
        return self._config

    def _first_config_path(self) -> Path:
        configs = sorted(self.path.glob("*_fold*/config.json"))
        if not configs:
            raise FileNotFoundError(f"No config.json under {self.path}")
        return configs[0]


def _parse_dir_timestamp(experiment_id: str) -> str:
    """Fallback timestamp extracted from the directory name, if present."""
    match = re.search(r"-(\d{8}-\d{6})", experiment_id)
    return match.group(1) if match else ""


def scan_run_dirs(results_dir: str | Path) -> list[RunRecord]:
    """Index every run directory that has at least one fold config."""
    results_path = Path(results_dir)
    records: list[RunRecord] = []
    for entry in sorted(results_path.iterdir()):
        if not entry.is_dir():
            continue
        configs = sorted(entry.glob("*_fold*/config.json"))
        if not configs:
            continue

        first_cfg_path = configs[0]
        with open(first_cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)

        n_splits = int(cfg.get("n_splits", 0) or 0)
        record = RunRecord(
            experiment_id=entry.name,
            path=entry,
            dataset=str(cfg.get("dataset") or ""),
            n_splits=n_splits,
            train_fraction=cfg.get("train_fraction"),
            curriculum_method=str(cfg.get("curriculum_method") or "biois_discrete"),
            timestamp=str(cfg.get("timestamp") or "") or _parse_dir_timestamp(entry.name),
        )

        # Discover modes and their fold completeness from subdirectory names.
        for sub in entry.iterdir():
            match = _FOLD_RE.match(sub.name)
            if not match or not sub.is_dir():
                continue
            mode = match.group("mode")
            record.modes.add(mode)
            if (sub / "phase_metrics.csv").exists():
                record.complete_modes[mode] = record.complete_modes.get(mode, 0) + 1

        records.append(record)
    return records


def _config_mismatches(
    cfg: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Return ``{key: (found, wanted)}`` for every expected field that differs."""
    return {
        key: (cfg.get(key), wanted)
        for key, wanted in expected.items()
        if cfg.get(key) != wanted
    }


def _matches_method(
    record: RunRecord,
    method,
    n_splits: int,
    expected_config: Mapping[str, Any] | None = None,
) -> bool:
    if method.mode not in record.complete_modes:
        return False
    if record.n_splits != n_splits:
        return False
    if record.train_fraction is not None and float(record.train_fraction) < 1.0:
        return False
    if record.curriculum_method != method.curriculum_method:
        return False
    cfg = record.config()
    # Encoder and batch size are not part of ``method.match`` because they are
    # supposed to be constant across a table. Enforce that here so a mixed
    # cell can never be resolved silently.
    if expected_config and _config_mismatches(cfg, expected_config):
        return False
    for key, expected in method.match.items():
        if cfg.get(key) != expected:
            return False
    return True


def resolve_latest(
    results_dir: str | Path,
    table: TableSpec,
    n_splits: int = 5,
    expected_config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return one row per ``(dataset, method)`` cell with its latest run dir.

    ``expected_config`` lists config fields every run must agree on (defaults to
    :data:`~bio_is_curriculum.results.paper_tables.REQUIRED_CONFIG`). Runs that
    disagree are dropped, and a warning names them, so a table cell is never
    filled from a run whose encoder or batch size differs from the rest. Pass an
    empty mapping to disable the check and resolve legacy runs.
    """
    if expected_config is None:
        expected_config = REQUIRED_CONFIG
    records = scan_run_dirs(results_dir)
    rows: list[dict] = []
    for dataset in table.datasets:
        for method in table.methods:
            candidates = [
                r
                for r in records
                if r.dataset == dataset
                and _matches_method(r, method, n_splits, expected_config)
            ]
            # Runs that matched everything except the shared config, reported so
            # the exclusion is visible rather than looking like "no run exists".
            rejected = [
                r
                for r in records
                if r.dataset == dataset
                and not _matches_method(r, method, n_splits, expected_config)
                and _matches_method(r, method, n_splits, None)
            ]
            if rejected:
                shown = rejected[:5]
                detail = "; ".join(
                    f"{r.experiment_id} {_config_mismatches(r.config(), expected_config)}"
                    for r in shown
                )
                if len(rejected) > len(shown):
                    detail += f"; (+{len(rejected) - len(shown)} more)"
                warnings.warn(
                    f"[{table.name}/{dataset}/{method.name}] {len(rejected)} run dir(s) "
                    f"excluded for config mismatch against {dict(expected_config)}: {detail}",
                    stacklevel=2,
                )
            if not candidates:
                rows.append(
                    {
                        "table": table.name,
                        "dataset": dataset,
                        "method": method.name,
                        "mode": method.mode,
                        "curriculum_method": method.curriculum_method,
                        "experiment_id": None,
                        "path": None,
                        "timestamp": "",
                        "n_folds": 0,
                    }
                )
                continue
            # Newest by recorded timestamp; tie-break on directory name order.
            best = max(candidates, key=lambda r: (r.timestamp, r.experiment_id))
            rows.append(
                {
                    "table": table.name,
                    "dataset": dataset,
                    "method": method.name,
                    "mode": method.mode,
                    "curriculum_method": method.curriculum_method,
                    "experiment_id": best.experiment_id,
                    "path": str(best.path),
                    "timestamp": best.timestamp,
                    "n_folds": best.complete_modes.get(method.mode, 0),
                }
            )
    return pd.DataFrame(rows)


def identify_latest(
    results_dir: str | Path,
    tables: list[TableSpec] | None = None,
    n_splits: int = 5,
    expected_config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Resolve latest runs for all paper tables into a single DataFrame."""
    frames = [
        resolve_latest(results_dir, table, n_splits=n_splits, expected_config=expected_config)
        for table in (tables or PAPER_TABLES)
    ]
    return pd.concat(frames, ignore_index=True)


def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Map each paper-table cell to its most recent result directory."
    )
    parser.add_argument("--results-dir", default="results", help="Base results directory.")
    parser.add_argument(
        "--output", default=None, help="Optional CSV path to write the mapping."
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Expected CV folds.")
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help=(
            "Resolve cells even when a run's hf_model/batch_size differs from "
            "REQUIRED_CONFIG. Use only to inspect legacy results; the paper "
            "tables require matched configs."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    # An empty mapping disables the check; ``None`` would be re-defaulted to
    # REQUIRED_CONFIG by resolve_latest.
    expected = {} if args.allow_config_mismatch else REQUIRED_CONFIG
    df = identify_latest(
        args.results_dir, n_splits=args.n_splits, expected_config=expected
    )
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved mapping to: {os.path.abspath(args.output)}")
    else:
        pd.set_option("display.width", 200)
        pd.set_option("display.max_rows", None)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
