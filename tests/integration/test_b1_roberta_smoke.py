"""Integration smoke test for Bengio b1 with RoBERTa."""

import pytest

from bio_is_curriculum.config.schema import ExperimentConfig
from bio_is_curriculum.pipeline.runner import run_experiment


@pytest.mark.integration
@pytest.mark.slow
def test_b1_roberta_webkb_smoke(tmp_path):
    cfg = ExperimentConfig(
        dataset="webkb",
        fold=0,
        n_splits=10,
        baseline=1,
        model="roberta",
        b1_easy_fraction=0.5,
        epochs_per_phase=1,
        batch_size=8,
        eval_batch_size=16,
        max_length=128,
        lr=2e-5,
        results_dir=str(tmp_path),
        experiment_id="test-b1-smoke",
    )
    metrics = run_experiment(cfg)
    assert metrics  # completes without error
