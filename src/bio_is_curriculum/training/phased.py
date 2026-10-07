"""Phased curriculum trainer (wraps model.fit_stage)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from bio_is_curriculum.results.metrics import build_phase_metrics_row

if TYPE_CHECKING:
    from bio_is_curriculum.models.base import CurriculumModel
    from bio_is_curriculum.results.recorder import RunRecorder


def eval_single_stage(
    model: CurriculumModel,
    X_eval,
    y_test,
    recorder: RunRecorder | None,
    *,
    phase: str = "full",
    train_time: float = float("nan"),
    hard_slice_quantile: float = 0.8,
    n_train_instances: int | None = None,
):
    t0 = time.perf_counter()
    proba = model.predict_proba(X_eval)
    preds = __import__("numpy").argmax(proba, axis=1)
    pred_time = time.perf_counter() - t0
    training_stats = model.get_training_stats() if hasattr(model, "get_training_stats") else {}
    row = build_phase_metrics_row(
        phase=phase,
        y_true=y_test,
        y_pred=preds,
        proba=proba,
        n_iter=model.n_iter,
        train_time_s=train_time,
        pred_time_s=pred_time,
        hard_slice_quantile=hard_slice_quantile,
        training_stats=training_stats,
        n_train_instances=n_train_instances,
    )
    if recorder is not None:
        recorder.log_phase(row)
    return preds, proba, row
