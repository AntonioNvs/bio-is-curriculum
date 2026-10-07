"""BIOIS weak-classifier signals (redundancy, entropy, margin, and noise risk)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from bio_is_curriculum.signals.heuristics import length_difficulty
from bio_is_curriculum.signals.oracle_margin import multiclass_margins


def bounded_entropy(probas: np.ndarray) -> np.ndarray:
    """Per-sample Shannon entropy normalized by log(n_classes)."""
    probas = np.asarray(probas, dtype=np.float64)
    ent = np.array([stats.entropy(p) for p in probas], dtype=np.float64)
    n_classes = probas.shape[1]
    if n_classes <= 1:
        return np.zeros_like(ent)
    cap = float(np.log(n_classes))
    if cap <= 0.0:
        return np.zeros_like(ent)
    return np.clip(ent / cap, 0.0, 1.0)


def per_class_rank_normalize(signal: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Map each class slice to [0, 1] via stable percentile ranks."""
    signal = np.asarray(signal, dtype=np.float64)
    y = np.asarray(y)
    out = np.zeros_like(signal)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        if idx.size == 0:
            continue
        if idx.size == 1:
            out[idx[0]] = 0.0
            continue
        order = np.argsort(signal[idx], kind="stable")
        ranks = np.linspace(0.0, 1.0, idx.size, endpoint=True)
        out[idx[order]] = ranks
    return out


def margin_difficulty(probas: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Label-aware difficulty from multiclass margin (higher = harder)."""
    margins = multiclass_margins(probas, y)
    return per_class_rank_normalize(-margins, y)


def redundancy_scores(probas: np.ndarray, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """True-class confidence for correct predictions; zero for mistakes."""
    y_arr = np.asarray(y)
    pred = np.asarray(pred)
    probas = np.asarray(probas, dtype=np.float64)
    r = np.zeros(len(y_arr), dtype=np.float64)
    correct = pred == y_arr
    if np.any(correct):
        r[correct] = probas[correct, y_arr[correct]]
    return per_class_rank_normalize(r, y_arr)


def noise_scores(selector, y) -> np.ndarray:
    """Deterministic noise risk from weak-classifier predictions."""
    if not hasattr(selector, "_probaEveryone"):
        raise ValueError(
            "selector lacks _probaEveryone; ensure BIOIS.fit was called first."
        )

    probas = selector._probaEveryone
    pred = selector._pred
    y_arr = np.asarray(y)
    e = bounded_entropy(probas)

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
    pred = selector._pred
    y_arr = np.asarray(y)

    e = per_class_rank_normalize(bounded_entropy(probas), y_arr)
    r = redundancy_scores(probas, y_arr, pred)

    noise = np.zeros_like(e)
    wrong = pred != y_arr
    noise[wrong] = 1.0 - bounded_entropy(probas)[wrong]

    return r, e, noise


def extract_biois_curriculum_signals(
    selector,
    y,
    texts: list[str] | None = None,
    *,
    margin_weight: float = 0.6,
    entropy_weight: float = 0.4,
    length_weight: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite BIO-IS curriculum difficulty with optional length prior."""
    if not hasattr(selector, "_probaEveryone"):
        raise ValueError(
            "selector lacks _probaEveryone; ensure BIOIS.fit was called first."
        )

    probas = selector._probaEveryone
    pred = selector._pred
    y_arr = np.asarray(y)

    entropy = per_class_rank_normalize(bounded_entropy(probas), y_arr)
    margin_d = margin_difficulty(probas, y_arr)
    bio_difficulty = margin_weight * margin_d + entropy_weight * entropy

    if texts is not None and length_weight > 0.0:
        length_rank = per_class_rank_normalize(length_difficulty(texts), y_arr)
        schedule = (1.0 - length_weight) * bio_difficulty + length_weight * length_rank
    else:
        schedule = bio_difficulty

    noise = np.zeros_like(schedule)
    wrong = pred != y_arr
    noise[wrong] = 1.0 - bounded_entropy(probas)[wrong]
    schedule_eff = np.maximum(schedule, noise)

    r = redundancy_scores(probas, y_arr, pred)
    return r, schedule_eff, noise
