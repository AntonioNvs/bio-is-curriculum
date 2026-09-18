"""Integration smoke test for self-adaptive PMD with ModernBERT."""

import pytest

from bio_is_curriculum.config.schema import ExperimentConfig
from bio_is_curriculum.pipeline.runner import run_experiment


@pytest.mark.integration
@pytest.mark.slow
def test_self_adaptive_pmd_modernbert_webkb_smoke(tmp_path):
    cfg = ExperimentConfig(
        dataset="webkb",
        fold=0,
        n_splits=10,
        mode="cl",
        curriculum_method="self_adaptive_pmd",
        model="modernbert",
        epochs=1,
        epochs_per_phase=1,
        batch_size=8,
        eval_batch_size=16,
        max_length=128,
        lr=5e-5,
        train_fraction=0.05,
        sa_score_batch_size=8,
        results_dir=str(tmp_path),
        experiment_id="test-sa-pmd-smoke",
    )
    metrics = run_experiment(cfg)
    assert metrics
    assert "macro_f1" in metrics
    run_dir = tmp_path / "test-sa-pmd-smoke" / "cl_fold0"
    assert (run_dir / "self_adaptive_verbalizers.json").exists()
    assert (run_dir / "self_adaptive_scores.csv").exists()
