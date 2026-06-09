"""Helpers para calcular metricas de avaliacao de forma consistente."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score


def hard_slice_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    quantile: float,
) -> float:
    """Calcula macro-F1 no top quantile de entropia preditiva."""
    if len(y_true) == 0 or len(proba) == 0:
        return float("nan")

    ent = np.array([stats.entropy(p) for p in proba], dtype=np.float64)
    threshold = np.quantile(ent, quantile)
    mask_hard = ent >= threshold
    if mask_hard.sum() == 0:
        return float("nan")
    return float(f1_score(y_true[mask_hard], y_pred[mask_hard], average="macro"))


def build_phase_metrics_row(
    *,
    phase: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    n_iter: int,
    train_time_s: float,
    pred_time_s: float,
    hard_slice_quantile: float,
    training_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Monta uma linha padrao de phase_metrics.csv."""
    stats_row = training_stats or {}
    return {
        "phase": phase,
        "n_samples": int(len(y_true)),
        "n_iter": int(n_iter),
        "train_time_s": float(train_time_s),
        "pred_time_s": float(pred_time_s),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "hard_slice_quantile": float(hard_slice_quantile),
        "hard_slice_macro_f1": hard_slice_macro_f1(y_true, y_pred, proba, hard_slice_quantile),
        "avg_seq_len": float(stats_row.get("avg_seq_len", float("nan"))),
        "compute_proxy": float(stats_row.get("compute_proxy", float("nan"))),
        "best_val_macro_f1": float(stats_row.get("best_val_macro_f1", float("nan"))),
        "best_val_epoch": float(stats_row.get("best_val_epoch", float("nan"))),
        "steps_to_best_val": float(stats_row.get("steps_to_best_val", float("nan"))),
    }
