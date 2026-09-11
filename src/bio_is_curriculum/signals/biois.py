"""BIOIS weak-classifier signals (redundancy, entropy, and noise risk)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _normalized_entropy(probas: np.ndarray) -> np.ndarray:
    """Per-sample Shannon entropy min-max normalized to [0, 1]."""
    e = np.array([stats.entropy(p) for p in probas], dtype=np.float64)
    e_range = e.max() - e.min()
    if e_range > 0:
        return (e - e.min()) / e_range
    return np.zeros_like(e)


def noise_scores(selector, y) -> np.ndarray:
    """Deterministic noise risk from weak-classifier predictions.

    Misclassified samples with low entropy (confident mistakes) receive the
    highest scores; correctly predicted samples receive zero.
    """
    if not hasattr(selector, "_probaEveryone"):
        raise ValueError(
            "selector lacks _probaEveryone; ensure BIOIS.fit was called first."
        )

    probas = selector._probaEveryone
    pred = selector._pred
    y_arr = np.asarray(y)
    e = _normalized_entropy(probas)

    noise = np.zeros(len(y_arr), dtype=np.float64)
    wrong = pred != y_arr
    noise[wrong] = 1.0 - e[wrong]
    return noise


def extract_biois_signals(selector, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive normalized (r, e, noise) from a fitted BIOIS selector."""
    if not hasattr(selector, "_probaEveryone"):
        raise ValueError(
            "selector lacks _probaEveryone; ensure BIOIS.fit was called first."
        )

    probas = selector._probaEveryone
    y_proba_pred = selector._y_proba_of_pred
    pred = selector._pred
    y_arr = np.asarray(y)

    e = _normalized_entropy(probas)

    r = np.array(y_proba_pred, dtype=np.float64, copy=True)
    r[pred != y_arr] = 0.0
    r_range = r.max() - r.min()
    if r_range > 0:
        r = (r - r.min()) / r_range
    else:
        r = np.zeros_like(r)

    noise = np.zeros_like(e)
    wrong = pred != y_arr
    noise[wrong] = 1.0 - e[wrong]

    return r, e, noise
