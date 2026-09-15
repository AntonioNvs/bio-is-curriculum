"""Shared discrete curriculum schedule (clean -> diverse -> hard)."""

from __future__ import annotations

import numpy as np

from bio_is_curriculum.curriculum.class_balance import per_class_low_quantile_mask
from bio_is_curriculum.curriculum.orchestrator import BIOISCurriculumBase


class DiscreteCurriculumBase(BIOISCurriculumBase):
    """Three-phase cumulative discrete curriculum driven by a difficulty signal."""

    REQUIRES_BIOIS: bool = True
    PHASE_NAMES = ("clean", "diverse", "hard")
    METHOD_ID: str = "discrete_base"

    def __init__(
        self,
        model=None,
        beta: float = 0.5,
        q_low: float = 0.3,
        q_mid: float = 0.6,
        q_high: float = 0.95,
        hard_slice_quantile: float = 0.8,
        r_cap: float = 0.5,
        random_state: int = 42,
        phase_max_lengths: tuple[int, int, int] | None = None,
    ):
        super().__init__(
            model=model,
            beta=beta,
            hard_slice_quantile=hard_slice_quantile,
            random_state=random_state,
        )
        self.q_low = q_low
        self.q_mid = q_mid
        self.q_high = q_high
        self.r_cap = r_cap
        self.phase_max_lengths = (
            tuple(phase_max_lengths)
            if phase_max_lengths is not None
            else (96, 160, 256)
        )
        self._y_build: np.ndarray | None = None

    def _build_phases(self, r, e):
        """Build cumulative phase indices and weights from difficulty signal ``e``."""
        y = self._y_build
        if y is None:
            raise RuntimeError("_y_build not set; call fit() first.")

        idx_all = np.arange(len(e))
        masks = (
            per_class_low_quantile_mask(e, y, self.q_low),
            per_class_low_quantile_mask(e, y, self.q_mid),
            per_class_low_quantile_mask(e, y, self.q_high),
        )

        e_mid_per_idx = np.zeros(len(e), dtype=np.float64)
        e_high_per_idx = np.zeros(len(e), dtype=np.float64)
        for cls in np.unique(y):
            cls_idx = np.flatnonzero(y == cls)
            e_mid_per_idx[cls_idx] = np.quantile(e[cls_idx], self.q_mid)
            e_high_per_idx[cls_idx] = np.quantile(e[cls_idx], self.q_high)

        phase_max_by_name = dict(zip(self.PHASE_NAMES, self.phase_max_lengths))
        phases = []
        for name, mask in zip(self.PHASE_NAMES, masks):
            indices = idx_all[mask]
            weights = np.ones(len(indices), dtype=np.float64)
            if name == "hard":
                hard_local = (e[indices] > e_mid_per_idx[indices]) & (
                    e[indices] <= e_high_per_idx[indices]
                )
                r_local = np.minimum(r[indices][hard_local], self.r_cap)
                weights[hard_local] = 1.0 - self.beta * r_local
                weights = np.clip(weights, 1e-6, None)
            phases.append(
                {
                    "name": name,
                    "indices": indices,
                    "weights": weights,
                    "max_length": phase_max_by_name.get(name),
                }
            )

        return phases

    def fit(self, selector, X, y, **kwargs):
        self._y_build = np.asarray(y)
        return super().fit(selector, X, y, **kwargs)
