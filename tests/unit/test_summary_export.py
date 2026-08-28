"""Tests for manifest-driven summary export."""

from __future__ import annotations

import json

import pandas as pd

from bio_is_curriculum.results.summary_export import export_from_manifest


def _write_summary_csv(
    exp_dir,
    *,
    mode: str = "cl",
    macro_f1: float = 0.75,
    curriculum_method: str = "biois_discrete",
    dataset: str | None = None,
) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "mode": mode,
                "metric": "macro_f1",
                "mean": macro_f1,
                "ci_95_low": macro_f1 - 0.01,
                "ci_95_high": macro_f1 + 0.01,
            }
        ]
    )
    df.to_csv(exp_dir / "summary.csv", index=False)
    fold_dir = exp_dir / f"{mode}_fold0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ds = dataset or exp_dir.name.split("-")[0]
    (fold_dir / "config.json").write_text(
        json.dumps({"dataset": ds, "mode": mode, "curriculum_method": curriculum_method}),
        encoding="utf-8",
    )


def test_export_from_manifest_long_table(tmp_path):
    exp_a = tmp_path / "results" / "webkb-10cv-20260101-120000_biois"
    exp_b = tmp_path / "results" / "webkb-10cv-20260101-120000_length"
    _write_summary_csv(exp_a, macro_f1=0.80, curriculum_method="biois_discrete")
    _write_summary_csv(exp_b, macro_f1=0.70, curriculum_method="length_discrete")

    manifest = {
        "event_description": "ablations",
        "timestamp": "20260101-120000",
        "summary": {"layout": "long_table", "metrics": ["macro_f1"], "datasets": None},
        "runs": [
            {"path": str(exp_a)},
            {"path": str(exp_b)},
        ],
    }
    manifest_path = tmp_path / "results" / "experiments" / "ablations_20260101-120000.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    xlsx_path, csv_path = export_from_manifest(manifest_path)
    assert xlsx_path.exists()
    assert csv_path.exists()
    csv_df = pd.read_csv(csv_path)
    assert len(csv_df) == 2
    assert set(csv_df["metric"]) == {"macro_f1"}


def test_export_from_manifest_compare_by_dataset(tmp_path):
    exp_a = tmp_path / "results" / "webkb-10cv-20260101-120000_biois"
    _write_summary_csv(exp_a, macro_f1=0.80)

    manifest = {
        "event_description": "compare",
        "timestamp": "20260101-120000",
        "summary": {"layout": "compare_by_dataset", "metrics": ["macro_f1"], "datasets": ["webkb"]},
        "runs": [{"path": str(exp_a)}],
    }
    manifest_path = tmp_path / "results" / "experiments" / "compare_20260101-120000.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    xlsx_path, csv_path = export_from_manifest(manifest_path)
    assert xlsx_path.exists()
    assert csv_path.exists()
    sheets = pd.ExcelFile(xlsx_path).sheet_names
    assert "webkb" in sheets
