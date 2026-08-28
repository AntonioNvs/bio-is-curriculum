"""Baseline 1 — Margin-paced Curriculum Learning (Bengio et al. 2009).

Two-stage discrete curriculum (easy subset → full target) ordered by
multiclass margin from an OOF TF-IDF logistic regression oracle proxy
(§4.2). No BIOIS instance selection, redundancy, or entropy weighting.

Reference
---------
Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).
Curriculum Learning. ICML 2009.
https://doi.org/10.1145/1553374.1553380
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from bio_is_curriculum.curriculum.orchestrator import BIOISCurriculumBase
from bio_is_curriculum.signals.oracle_margin import (
    easy_indices_by_margin,
    multiclass_margins,
    oof_lr_probas,
)

if TYPE_CHECKING:
    from bio_is_curriculum.results.recorder import RunRecorder


class Baseline1(BIOISCurriculumBase):
    INDEX = 1
    NAME = "Margin-paced CL (Bengio 2009)"
    REFERENCE = (
        "Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). "
        "Curriculum Learning. ICML 2009. "
        "https://doi.org/10.1145/1553374.1553380"
    )
    TRAINER_KIND = "phased"
    REQUIRES_BIOIS = False

    PHASE_NAMES = ("easy", "target")
    DEFAULT_EASY_FRACTION = 0.5

    def __init__(
        self,
        model=None,
        easy_fraction: float = DEFAULT_EASY_FRACTION,
        use_global_quantile: bool = True,
        hard_slice_quantile: float = 0.8,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            model=model,
            beta=0.0,
            hard_slice_quantile=hard_slice_quantile,
            random_state=random_state,
        )
        self.easy_fraction = easy_fraction
        self.use_global_quantile = use_global_quantile
        self._y_build: np.ndarray | None = None
        self._X_build = None

    def _build_phases(self, r, e):
        """Satisfy abstract base; ``fit()`` calls ``_build_phase_list`` directly."""
        return self._build_phase_list(np.asarray(r))

    def _build_phase_list(self, margin: np.ndarray) -> list[dict]:
        """Two cumulative phases: easy subset then full target set."""
        y = self._y_build
        if y is None:
            raise RuntimeError("_y_build not set; call fit() first.")

        idx_all = np.arange(len(margin))
        easy_idx = easy_indices_by_margin(
            margin,
            y,
            self.easy_fraction,
            use_global=self.use_global_quantile,
        )

        phases = []
        for name, indices in (
            ("easy", easy_idx),
            ("target", idx_all),
        ):
            weights = np.ones(len(indices), dtype=np.float64)
            phases.append({"name": name, "indices": indices, "weights": weights})
        return phases

    def _extract_signals(self, selector, y):
        raise NotImplementedError("Baseline1 computes margins directly from X.")

    def fit(
        self,
        selector,
        X,
        y,
        X_test=None,
        y_test=None,
        X_val=None,
        y_val=None,
        X_text=None,
        X_val_text=None,
        X_test_text=None,
        recorder: Optional["RunRecorder"] = None,
    ):
        """Score OOF LR margins on TF-IDF X, build 2-phase schedule, train."""
        del selector  # standalone scorer — no BIOIS selector
        self._y_build = np.asarray(y)
        self._X_build = X

        t0_signals = time.perf_counter()
        probas = oof_lr_probas(X, y, random_state=self.random_state)
        margin = multiclass_margins(probas, y)
        self._coverage_score = margin
        signal_time = time.perf_counter() - t0_signals

        t0_phases = time.perf_counter()
        self.phases_ = self._build_phase_list(margin)
        phases_time = time.perf_counter() - t0_phases

        if recorder is not None:
            recorder.log_timing("b1_margin_score_time_s", signal_time)
            recorder.log_timing("cl_phase_build", phases_time)

        self.history_ = self._run_phase_loop(
            self.phases_,
            X,
            y,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
            X_text=X_text,
            X_val_text=X_val_text,
            X_test_text=X_test_text,
            recorder=recorder,
        )
        return self
