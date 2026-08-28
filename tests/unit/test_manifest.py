"""Tests for experiment manifest read/write."""

from __future__ import annotations

import json

from bio_is_curriculum.config.loader import load_experiment_spec
from bio_is_curriculum.config.schema import BatchExperimentConfig, ExperimentConfig, SummaryConfig
from bio_is_curriculum.results.manifest import (
    ExperimentManifest,
    build_run_entry,
    load_manifest,
    manifest_filename,
    manifest_path,
    resolve_event_description,
    resolve_summary_config,
    sanitize_event_name,
    write_manifest,
)


def test_sanitize_event_name():
    assert sanitize_event_name("full cv multi!") == "full_cv_multi"
    assert sanitize_event_name("  ") == "experiment"


def test_manifest_filename():
    assert manifest_filename("curriculum_ablations_multi", "20260828-014706") == (
        "curriculum_ablations_multi_20260828-014706.json"
    )


def test_write_and_load_manifest(tmp_path):
    manifest = ExperimentManifest(
        event_description="smoke_test",
        timestamp="20260101-120000",
        config_path="experiments/smoke.yaml",
        started_at="2026-01-01T12:00:00Z",
        finished_at="2026-01-01T12:05:00Z",
        summary={"layout": "long_table", "metrics": ["macro_f1"], "datasets": None},
        runs=[],
    )
    path = write_manifest(manifest, tmp_path)
    assert path == tmp_path / "experiments" / "smoke_test_20260101-120000.json"
    loaded = load_manifest(path)
    assert loaded["event_description"] == "smoke_test"
    assert loaded["summary"]["layout"] == "long_table"


def test_build_run_entry():
    batch = BatchExperimentConfig(
        dataset="webkb",
        n_splits=10,
        modes=["cl"],
        experiment_id="webkb-10cv-20260101-120000_biois_discrete",
        run=ExperimentConfig(curriculum_method="biois_discrete", results_dir="results"),
    )
    entry = build_run_entry(
        batch,
        "webkb-10cv-20260101-120000_biois_discrete",
        status="ok",
        results_dir="results",
    )
    assert entry.dataset == "webkb"
    assert entry.path == "results/webkb-10cv-20260101-120000_biois_discrete"
    assert entry.status == "ok"


def test_resolve_event_description_from_campaign_name(tmp_path):
    path = tmp_path / "my_campaign.yaml"
    path.write_text(
        "campaign:\n  name: custom_event\n  datasets:\n    webkb: {}\n  jobs:\n    - modes: [raw]\n",
        encoding="utf-8",
    )
    spec = load_experiment_spec(str(path))
    assert resolve_event_description(spec) == "custom_event"


def test_resolve_summary_config_auto_layout():
    from bio_is_curriculum.config.schema import CampaignSpec, ExperimentSpec

    single = ExperimentSpec(config_path="experiments/smoke.yaml", summary=None)
    cfg_single = resolve_summary_config(single, 1)
    assert cfg_single.resolve_layout(1) == "long_table"

    multi = ExperimentSpec(
        config_path="experiments/campaign.yaml",
        campaign=CampaignSpec(
            summary=SummaryConfig(layout="compare_by_dataset"),
        ),
    )
    cfg_multi = resolve_summary_config(multi, 4)
    assert cfg_multi.resolve_layout(4) == "compare_by_dataset"


def test_load_campaign_summary_block(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(
        """
campaign:
  name: ablations
  summary:
    layout: compare_by_dataset
    metrics: [macro_f1]
  datasets:
    webkb: {}
  jobs:
    - modes: [cl]
""",
        encoding="utf-8",
    )
    spec = load_experiment_spec(str(path))
    assert spec.campaign is not None
    assert spec.campaign.name == "ablations"
    assert spec.campaign.summary is not None
    assert spec.campaign.summary.layout == "compare_by_dataset"

    resolved = resolve_summary_config(spec, 2)
    assert resolved.metrics == ["macro_f1"]

    out = manifest_path(tmp_path, "ablations", "20260101-120000")
    assert out.name == "ablations_20260101-120000.json"

    manifest = ExperimentManifest(
        event_description="ablations",
        timestamp="20260101-120000",
        config_path=str(path),
        summary=resolved.to_dict(),
    )
    written = write_manifest(manifest, tmp_path)
    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["summary"]["metrics"] == ["macro_f1"]
