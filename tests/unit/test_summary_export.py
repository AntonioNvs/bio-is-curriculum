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
    curriculum_q: list[float] | None = None,
    curriculum_beta: float | None = None,
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
    config = {"dataset": ds, "mode": mode, "curriculum_method": curriculum_method}
    if curriculum_q is not None:
        config["curriculum_q"] = curriculum_q
    if curriculum_beta is not None:
        config["curriculum_beta"] = curriculum_beta
    (fold_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


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
    exp_dir = tmp_path / "results" / "experiments" / "ablations_20260101-120000"
    exp_dir.mkdir(parents=True)
    manifest_path = exp_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    xlsx_path, csv_path = export_from_manifest(exp_dir)
    assert xlsx_path == exp_dir / "summary.xlsx"
    assert csv_path == exp_dir / "summary.csv"
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
    exp_dir = tmp_path / "results" / "experiments" / "compare_20260101-120000"
    exp_dir.mkdir(parents=True)
    (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    xlsx_path, csv_path = export_from_manifest(exp_dir)
    assert xlsx_path.exists()
    assert csv_path.exists()
    sheets = pd.ExcelFile(xlsx_path).sheet_names
    assert "webkb" in sheets


def test_export_from_manifest_distinguishes_cl_param_variants(tmp_path):
    exp_weighted = tmp_path / "results" / "webkb-10cv-20260101-120000_q03-06_weighted"
    exp_tight = tmp_path / "results" / "webkb-10cv-20260101-120000_q02-05_weighted"
    exp_unweighted = tmp_path / "results" / "webkb-10cv-20260101-120000_q03-06_unweighted"
    _write_summary_csv(
        exp_weighted,
        macro_f1=0.73,
        curriculum_q=[0.3, 0.6, 0.95],
        curriculum_beta=0.5,
    )
    _write_summary_csv(
        exp_tight,
        macro_f1=0.72,
        curriculum_q=[0.2, 0.5, 0.95],
        curriculum_beta=0.5,
    )
    _write_summary_csv(
        exp_unweighted,
        macro_f1=0.74,
        curriculum_q=[0.3, 0.6, 0.95],
        curriculum_beta=0.0,
    )

    manifest = {
        "event_description": "cl_params_ablation_multi",
        "timestamp": "20260101-120000",
        "summary": {
            "layout": "compare_by_dataset",
            "metrics": ["macro_f1"],
            "datasets": ["webkb"],
        },
        "runs": [
            {"path": str(exp_weighted)},
            {"path": str(exp_tight)},
            {"path": str(exp_unweighted)},
        ],
    }
    exp_dir = tmp_path / "results" / "experiments" / "cl_params_ablation_multi_20260101-120000"
    exp_dir.mkdir(parents=True)
    (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    xlsx_path, csv_path = export_from_manifest(exp_dir)
    csv_df = pd.read_csv(csv_path)
    assert len(csv_df) == 3
    assert len(set(csv_df["method"])) == 3

    wide = pd.read_excel(xlsx_path, sheet_name="webkb")
    assert len(wide) == 3
