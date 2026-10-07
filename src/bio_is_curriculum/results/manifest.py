"""Experiment run manifests under results/experiments/<event>_<timestamp>/."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bio_is_curriculum.config.schema import (
    BatchExperimentConfig,
    DockerConfig,
    ExperimentSpec,
    SummaryConfig,
)

MANIFEST_SCHEMA_VERSION = "1"
_EXPERIMENTS_SUBDIR = "experiments"
MANIFEST_JSON_NAME = "manifest.json"
SUMMARY_CSV_NAME = "summary.csv"
SUMMARY_XLSX_NAME = "summary.xlsx"


@dataclass
class ManifestRun:
    experiment_id: str
    dataset: str
    n_splits: int
    modes: list[str]
    curriculum_method: str
    path: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentManifest:
    schema_version: str = MANIFEST_SCHEMA_VERSION
    event_description: str = ""
    timestamp: str = ""
    config_path: str = ""
    started_at: str = ""
    finished_at: str = ""
    docker: dict[str, Any] | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    runs: list[ManifestRun] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_description": self.event_description,
            "timestamp": self.timestamp,
            "config_path": self.config_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "docker": self.docker,
            "summary": self.summary,
            "runs": [run.to_dict() for run in self.runs],
        }


def sanitize_event_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "experiment"


def experiment_dir_name(event_description: str, timestamp: str) -> str:
    return f"{sanitize_event_name(event_description)}_{timestamp}"


def resolve_event_description(spec: ExperimentSpec) -> str:
    if spec.campaign is not None and spec.campaign.name:
        return sanitize_event_name(spec.campaign.name)
    return sanitize_event_name(Path(spec.config_path).stem)


def resolve_summary_config(spec: ExperimentSpec, num_runs: int) -> SummaryConfig:
    cfg = None
    if spec.campaign is not None and spec.campaign.summary is not None:
        cfg = spec.campaign.summary
    elif spec.summary is not None:
        cfg = spec.summary
    else:
        cfg = SummaryConfig()
    resolved = SummaryConfig(
        layout=cfg.resolve_layout(num_runs),
        metrics=list(cfg.metrics),
        datasets=cfg.datasets,
    )
    return resolved


def experiment_dir(
    results_dir: str | Path,
    event_description: str,
    timestamp: str,
) -> Path:
    return (
        Path(results_dir)
        / _EXPERIMENTS_SUBDIR
        / experiment_dir_name(event_description, timestamp)
    )


def manifest_path(
    results_dir: str | Path,
    event_description: str,
    timestamp: str,
) -> Path:
    return experiment_dir(results_dir, event_description, timestamp) / MANIFEST_JSON_NAME


def resolve_manifest_path(path: str | Path) -> Path:
    """Resolve a manifest JSON path from a file or experiment folder."""
    candidate = Path(path)
    if candidate.is_dir():
        manifest = candidate / MANIFEST_JSON_NAME
        if manifest.is_file():
            return manifest
        json_files = sorted(candidate.glob("*.json"))
        if len(json_files) == 1:
            return json_files[0]
        if json_files:
            raise ValueError(
                f"Ambiguous experiment folder (multiple JSON files): {candidate}"
            )
        raise FileNotFoundError(f"No manifest JSON found in: {candidate}")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Manifest path not found: {candidate}")


def build_run_entry(
    batch: BatchExperimentConfig,
    experiment_id: str,
    *,
    status: str,
    results_dir: str,
) -> ManifestRun:
    cfg = batch.run
    rel_path = f"{results_dir.rstrip('/')}/{experiment_id}"
    return ManifestRun(
        experiment_id=experiment_id,
        dataset=batch.dataset or cfg.dataset,
        n_splits=batch.n_splits,
        modes=list(batch.modes),
        curriculum_method=cfg.curriculum_method,
        path=rel_path,
        status=status,
    )


def write_manifest(manifest: ExperimentManifest, results_dir: str | Path) -> Path:
    path = manifest_path(results_dir, manifest.event_description, manifest.timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
        f.write("\n")
    return path


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path_obj = resolve_manifest_path(path)
    with manifest_path_obj.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest (expected object): {manifest_path_obj}")
    return data


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def docker_to_dict(docker: DockerConfig | None) -> dict[str, Any] | None:
    if docker is None:
        return None
    return {
        "image": docker.image,
        "gpu_id": docker.gpu_id,
        "cpus": docker.cpus,
        "memory": docker.memory,
    }
