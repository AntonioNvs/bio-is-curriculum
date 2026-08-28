"""Difficulty signal extractors shared by curriculum methods and baselines."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DifficultyScorer(Protocol):
    """Protocol for sample difficulty scoring."""

    def score(self, *args, **kwargs) -> np.ndarray:
        """Return per-sample difficulty scores (higher = harder)."""
        ...
