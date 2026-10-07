"""Utilitarios de balanceamento de classes para curriculum learning."""
from __future__ import annotations

from typing import Any

import numpy as np

# Alinhado a BIOIS_STRATKFOLD_SPLITS em cli.py.
RARE_CLASS_PIN_THRESHOLD = 5
CLASS_COVERAGE_MIN = 1


def class_counts(y: np.ndarray) -> dict[int, int]:
    """Contagem por rotulo."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    return {int(c): int(k) for c, k in zip(classes, counts)}


def pin_rare_class_indices(
    y: np.ndarray,
    *,
    max_count: int = RARE_CLASS_PIN_THRESHOLD,
) -> np.ndarray:
    """Indices de todas as amostras em classes com contagem <= max_count."""
    y = np.asarray(y)
    counts = class_counts(y)
    pinned: list[int] = []
    for cls, cnt in counts.items():
        if cnt <= max_count:
            pinned.extend(np.flatnonzero(y == cls).tolist())
    if not pinned:
        return np.empty(0, dtype=int)
    return np.sort(np.unique(pinned))


def ensure_class_coverage(
    selected: np.ndarray,
    y: np.ndarray,
    *,
    score: np.ndarray | None = None,
    min_per_class: int = CLASS_COVERAGE_MIN,
) -> np.ndarray:
    """Garante pelo menos ``min_per_class`` amostras por classe presente em ``y``."""
    y = np.asarray(y)
    selected = np.asarray(selected, dtype=int)
    if selected.size == 0:
        selected = np.empty(0, dtype=int)

    classes_needed = np.unique(y)
    for cls in classes_needed:
        cls_int = int(cls)
        cls_idx = np.flatnonzero(y == cls)
        if cls_idx.size == 0:
            continue
        present = int(np.sum(y[selected] == cls))
        need = min_per_class - present
        if need <= 0:
            continue
        available = cls_idx[~np.isin(cls_idx, selected)]
        if available.size == 0:
            continue
        if score is not None:
            pick_order = available[np.argsort(-score[available], kind="stable")]
        else:
            pick_order = available
        to_add = pick_order[:need]
        selected = np.sort(np.append(selected, to_add.astype(int)))

    return selected


def balance_phase_indices(
    indices: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    score: np.ndarray | None = None,
    rare_threshold: int = RARE_CLASS_PIN_THRESHOLD,
    min_per_class: int = CLASS_COVERAGE_MIN,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Aplica pinning de classes raras e cobertura minima a um conjunto de fase."""
    y = np.asarray(y)
    indices = np.asarray(indices, dtype=int)
    if weights is None:
        weights = np.ones(len(indices), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    n_classes_total = int(np.unique(y).size)
    pinned = pin_rare_class_indices(y, max_count=rare_threshold)
    n_rare_classes_pinned = sum(
        1 for cnt in class_counts(y).values() if cnt <= rare_threshold
    )

    merged = np.sort(np.unique(np.concatenate([indices, pinned])) if pinned.size else indices)
    merged = ensure_class_coverage(
        merged, y, score=score, min_per_class=min_per_class,
    )

    weight_map = {int(idx): float(w) for idx, w in zip(indices, weights)}
    default_w = float(np.median(weights)) if weights.size else 1.0
    new_weights = np.array(
        [weight_map.get(int(idx), default_w) for idx in merged],
        dtype=np.float64,
    )

    present = set(np.unique(y[merged]).tolist())
    missing = sorted(set(np.unique(y).tolist()) - present)
    stats = {
        "n_train_samples": int(merged.size),
        "n_classes_present": int(len(present)),
        "n_classes_total": n_classes_total,
        "n_classes_missing": int(len(missing)),
        "n_rare_classes_pinned": int(n_rare_classes_pinned),
        "missing_classes": missing,
    }
    return merged, new_weights, stats


def per_class_low_quantile_mask(
    signal: np.ndarray,
    y: np.ndarray,
    q: float,
    *,
    rare_threshold: int = RARE_CLASS_PIN_THRESHOLD,
) -> np.ndarray:
    """Mascara cumulativa: por classe, menores ``q`` fracao de ``signal`` (faceis)."""
    signal = np.asarray(signal, dtype=np.float64)
    y = np.asarray(y)
    n = len(y)
    mask = np.zeros(n, dtype=bool)
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        if cls_idx.size <= rare_threshold:
            mask[cls_idx] = True
            continue
        thresh = np.quantile(signal[cls_idx], q)
        mask[cls_idx] = signal[cls_idx] <= thresh
    return mask


def per_class_high_quantile_mask(
    signal: np.ndarray,
    y: np.ndarray,
    q: float,
    *,
    rare_threshold: int = RARE_CLASS_PIN_THRESHOLD,
) -> np.ndarray:
    """Mascara cumulativa: por classe, maiores ``q`` fracao de ``signal`` (faceis)."""
    signal = np.asarray(signal, dtype=np.float64)
    y = np.asarray(y)
    n = len(y)
    mask = np.zeros(n, dtype=bool)
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        if cls_idx.size <= rare_threshold:
            mask[cls_idx] = True
            continue
        thresh = np.quantile(signal[cls_idx], 1.0 - q)
        mask[cls_idx] = signal[cls_idx] >= thresh
    return mask


def balanced_epoch_weights(
    y: np.ndarray,
    *,
    rare_threshold: int = RARE_CLASS_PIN_THRESHOLD,
) -> tuple[np.ndarray, int]:
    """Pesos inversamente proporcionais a frequencia de classe para um epoch."""
    y = np.asarray(y)
    n = len(y)
    counts = class_counts(y)
    inv = np.array([1.0 / counts[int(c)] for c in y], dtype=np.float64)
    inv /= inv.sum()
    n_classes = len(counts)
    # Garante pelo menos um sorteio por classe quando ha classes raras.
    num_samples = max(n, n_classes)
    return inv, num_samples
