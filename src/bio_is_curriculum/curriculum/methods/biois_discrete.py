"""BIOIS discrete curriculum: clean -> diverse -> hard using entropy signal."""
from __future__ import annotations

import numpy as np

from bio_is_curriculum.curriculum.methods.discrete_base import DiscreteCurriculumBase
from bio_is_curriculum.signals.biois import extract_biois_signals


class BIOISDiscreteCurriculum(DiscreteCurriculumBase):
    """Discrete curriculum over BIOIS redundancy, entropy, and noise signals."""

    REQUIRES_BIOIS = True
    METHOD_ID = "biois_discrete"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._noise_: np.ndarray | None = None

    def _extract_signals(self, selector, y):
        r, e, noise = extract_biois_signals(selector, y)
        self._noise_ = noise
        e_eff = np.maximum(e, noise)
        return r, e_eff

    def _build_phases(self, r, e):
        phases = super()._build_phases(r, e)
        if self._noise_ is None:
            return phases

        noise = self._noise_
        for phase in phases:
            indices = phase["indices"]
            weights = phase["weights"]
            phase["weights"] = np.clip(
                weights * (1.0 - self.beta * noise[indices]),
                1e-6,
                None,
            )
        return phases
