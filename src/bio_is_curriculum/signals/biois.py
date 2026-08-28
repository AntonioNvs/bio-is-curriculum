"""BIOIS weak-classifier signals (redundancy and entropy)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def extract_biois_signals(selector, y) -> tuple[np.ndarray, np.ndarray]:
    """Derive normalized (r, e) from a fitted BIOIS selector."""
    if not hasattr(selector, "_probaEveryone"):
        raise ValueError(
            "selector lacks _probaEveryone; ensure BIOIS.fit was called first."
        )

    probas = selector._probaEveryone
    y_proba_pred = selector._y_proba_of_pred
    pred = selector._pred
    y_arr = np.asarray(y)

    e = np.array([stats.entropy(p) for p in probas], dtype=np.float64)
    e_range = e.max() - e.min()
    e = (e - e.min()) / e_range if e_range > 0 else np.zeros_like(e)

    r = np.array(y_proba_pred, dtype=np.float64, copy=True)
    r[pred != y_arr] = 0.0
    r_range = r.max() - r.min()
    r = (r - r.min()) / r_range if r_range > 0 else np.zeros_like(r)

    return r, e
