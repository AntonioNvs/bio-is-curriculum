"""Nuclear-norm difficulty scoring for SPDCL (Zhang et al. 2022)."""

from __future__ import annotations

import numpy as np


def nuclear_norm(matrix: np.ndarray) -> float:
    """Compute tr(sqrt(E^T E)) = sum of singular values."""
    if matrix.size == 0:
        return 0.0
    s = np.linalg.svd(matrix, compute_uv=False)
    return float(np.sum(s))


def score_hidden_states(hidden_states: list[np.ndarray]) -> np.ndarray:
    """Per-sample nuclear norm from last-layer token matrices."""
    return np.array([nuclear_norm(h) for h in hidden_states], dtype=np.float64)


def spdcl_epoch_difficulty(
    current: np.ndarray,
    previous: np.ndarray | None,
    *,
    first_epoch: bool,
) -> np.ndarray:
    """SPDCL difficulty: absolute norm at t=1, delta norm afterwards."""
    if first_epoch or previous is None:
        return current.copy()
    return current - previous


class NuclearNormScorer:
    """Wraps linguistic (pretrain) and model-capacity (delta) SPDCL signals."""

    def __init__(self, initial_norms: np.ndarray | None = None):
        self.initial_norms = initial_norms
        self._previous: np.ndarray | None = None

    def score_pretrain(self, hidden_states: list[np.ndarray]) -> np.ndarray:
        """Linguistic difficulty from pretrained backbone (Algorithm 1, step 1)."""
        norms = score_hidden_states(hidden_states)
        self.initial_norms = norms
        self._previous = norms.copy()
        return norms

    def score_current(self, hidden_states: list[np.ndarray]) -> np.ndarray:
        """Absolute nuclear norms at the current training step."""
        return score_hidden_states(hidden_states)

    def score_delta(self, current: np.ndarray) -> np.ndarray:
        """Model-capacity difficulty: d_t - d_{t-1} (Algorithm 1, step 7-8)."""
        if self._previous is None:
            raise RuntimeError("Call score_pretrain before score_delta.")
        delta = spdcl_epoch_difficulty(current, self._previous, first_epoch=False)
        self._previous = current.copy()
        return delta

    def difficulty_for_epoch(self, current: np.ndarray, *, curriculum_epoch: int) -> np.ndarray:
        """Epoch 0 uses cached pretrain norms; later epochs use delta."""
        if curriculum_epoch == 0 and self.initial_norms is not None:
            return self.initial_norms.copy()
        return self.score_delta(current)
