"""BIOIS discrete curriculum: clean -> diverse -> hard using composite BIO-IS signals."""
from __future__ import annotations

import numpy as np

from bio_is_curriculum.curriculum.methods.discrete_base import DiscreteCurriculumBase
from bio_is_curriculum.curriculum.orchestrator import BIOISCurriculumBase
from bio_is_curriculum.signals.biois import extract_biois_curriculum_signals


class BIOISDiscreteCurriculum(DiscreteCurriculumBase):
    """Discrete curriculum over BIOIS margin, entropy, noise, and length prior."""

    REQUIRES_BIOIS = True
    METHOD_ID = "biois_discrete"
    PHASE_NAMES = ("clean", "diverse", "hard")

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
        margin_weight: float = 0.6,
        entropy_weight: float = 0.4,
        length_weight: float = 0.25,
        noise_weight_phases: tuple[str, ...] | None = None,
        phase_max_lengths: tuple[int, int, int] | None = None,
    ):
        super().__init__(
            model=model,
            beta=beta,
            q_low=q_low,
            q_mid=q_mid,
            q_high=q_high,
            hard_slice_quantile=hard_slice_quantile,
            r_cap=r_cap,
            random_state=random_state,
            phase_max_lengths=phase_max_lengths,
        )
        self.margin_weight = float(margin_weight)
        self.entropy_weight = float(entropy_weight)
        self.length_weight = float(length_weight)
        self.noise_weight_phases = (
            tuple(noise_weight_phases)
            if noise_weight_phases is not None
            else ("hard",)
        )
        self._noise_: np.ndarray | None = None
        self._texts_build: list[str] | None = None

    def _coverage_score_from_signals(self, r: np.ndarray, e: np.ndarray) -> np.ndarray:
        return -np.asarray(e, dtype=np.float64)

    def _extract_signals(self, selector, y):
        r, e, noise = extract_biois_curriculum_signals(
            selector,
            y,
            self._texts_build,
            margin_weight=self.margin_weight,
            entropy_weight=self.entropy_weight,
            length_weight=self.length_weight,
        )
        self._noise_ = noise
        return r, e

    def _build_phases(self, r, e):
        phases = super()._build_phases(r, e)

        if self._noise_ is None:
            return phases

        noise = self._noise_
        noise_phases = set(self.noise_weight_phases)
        for phase in phases:
            if phase["name"] not in noise_phases:
                continue
            indices = phase["indices"]
            weights = phase["weights"]
            phase["weights"] = np.clip(
                weights * (1.0 - self.beta * noise[indices]),
                1e-6,
                None,
            )
        return phases

    def fit(self, selector, X, y, **kwargs):
        self._texts_build = kwargs.get("X_text")
        self._y_build = np.asarray(y)
        return BIOISCurriculumBase.fit(self, selector, X, y, **kwargs)
