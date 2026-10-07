"""Heuristic difficulty signals (controls for future baselines)."""

from __future__ import annotations

import numpy as np


def length_difficulty(texts: list[str]) -> np.ndarray:
    """Token-count proxy: longer texts score as harder."""
    lengths = np.array([len(t.split()) for t in texts], dtype=np.float64)
    if lengths.max() == lengths.min():
        return np.zeros_like(lengths)
    return (lengths - lengths.min()) / (lengths.max() - lengths.min())
