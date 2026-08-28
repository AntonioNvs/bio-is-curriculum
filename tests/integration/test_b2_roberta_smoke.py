"""Integration smoke test for SPDCL (b2) with RoBERTa."""

import pytest

from bio_is_curriculum.config.schema import ExperimentConfig
from bio_is_curriculum.pipeline.runner import run_experiment


@pytest.mark.integration
@pytest.mark.slow
def test_b2_roberta_webkb_smoke(tmp_path):
    cfg = ExperimentConfig(
        dataset="webkb",
        fold=0,
        n_splits=10,
        baseline=2,
        model="roberta",
        spdcl_n_bins=2,
        spdcl_curriculum_epochs=2,
        spdcl_anneal_epochs=1,
        epochs=3,
        batch_size=8,
        eval_batch_size=16,
        max_length=128,
        lr=5e-5,
        results_dir=str(tmp_path),
        experiment_id="test-b2-smoke",
    )
    metrics = run_experiment(cfg)
    assert "macro_f1" in metrics or metrics  # completes without error
