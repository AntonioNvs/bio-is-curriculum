"""Per-sample training loss as a difficulty signal."""

from __future__ import annotations

import numpy as np


def sample_ce_loss(model, X, y, *, X_text=None) -> np.ndarray:
    """Cross-entropy per sample from model.predict_proba (higher = harder)."""
    X_input = X_text if X_text is not None else X
    probas = model.predict_proba(X_input)
    y_arr = np.asarray(y).astype(int)
    eps = 1e-12
    if probas.shape[1] > int(np.max(y_arr)):
        p_true = probas[np.arange(len(y_arr)), y_arr]
    elif hasattr(model, "_clf") and hasattr(model._clf, "classes_"):
        cls = np.asarray(model._clf.classes_).astype(int)
        col_by_class = {c: i for i, c in enumerate(cls.tolist())}
        cols = np.array(
            [col_by_class.get(int(lbl), -1) for lbl in y_arr], dtype=int
        )
        p_true = np.full(len(y_arr), eps, dtype=np.float64)
        valid = cols >= 0
        if np.any(valid):
            p_true[valid] = probas[np.arange(len(y_arr))[valid], cols[valid]]
    else:
        p_true = np.full(len(y_arr), eps, dtype=np.float64)
    p_true = np.clip(p_true, eps, 1.0)
    return -np.log(p_true)
