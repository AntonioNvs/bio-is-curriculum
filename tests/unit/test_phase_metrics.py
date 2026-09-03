"""Unit tests for phase metrics helpers."""

import numpy as np

from bio_is_curriculum.results.metrics import build_phase_metrics_row


def test_build_phase_metrics_row_train_and_test_counts():
    y = np.array([0, 1, 0])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    preds = np.argmax(proba, axis=1)
    row = build_phase_metrics_row(
        phase="clean",
        y_true=y,
        y_pred=preds,
        proba=proba,
        n_iter=10,
        train_time_s=1.0,
        pred_time_s=0.1,
        hard_slice_quantile=0.8,
        n_train_instances=42,
    )
    assert row["n_test_samples"] == 3
    assert row["n_samples"] == 3
    assert row["n_train_instances"] == 42
