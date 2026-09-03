"""Heuristic difficulty signals as discrete curriculum ablations."""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np

from bio_is_curriculum.curriculum.methods.discrete_base import DiscreteCurriculumBase
from bio_is_curriculum.signals.heuristics import length_difficulty
from bio_is_curriculum.signals.lexical import tfidf_difficulty
from bio_is_curriculum.signals.loss import sample_ce_loss

if TYPE_CHECKING:
    from bio_is_curriculum.results.recorder import RunRecorder


class HeuristicDiscreteCurriculum(DiscreteCurriculumBase):
    """Discrete curriculum with an alternative static difficulty signal (r=0)."""

    REQUIRES_BIOIS = False

    _X_build: object = None
    _texts: list[str] | None = None

    @abstractmethod
    def _compute_difficulty(self) -> np.ndarray:
        """Return per-sample difficulty (higher = harder)."""

    def _extract_signals(self, selector, y):
        d = self._compute_difficulty()
        return np.zeros_like(d), d

    def fit(self, selector, X, y, **kwargs):
        self._X_build = X
        self._texts = kwargs.get("X_text")
        return super().fit(selector, X, y, **kwargs)


class LengthDiscreteCurriculum(HeuristicDiscreteCurriculum):
    """Length-based difficulty ablation (Platanios et al., 2019)."""

    METHOD_ID = "length_discrete"

    def _compute_difficulty(self) -> np.ndarray:
        if self._texts is None:
            raise ValueError(
                "length_discrete requires raw texts (use model: modernbert)."
            )
        return length_difficulty(self._texts)


class TfidfDiscreteCurriculum(HeuristicDiscreteCurriculum):
    """TF-IDF rank difficulty ablation (Soviany et al., 2022 proxy)."""

    METHOD_ID = "tfidf_discrete"

    def _compute_difficulty(self) -> np.ndarray:
        if self._X_build is None:
            raise ValueError("tfidf_discrete requires TF-IDF feature matrix X.")
        return tfidf_difficulty(self._X_build)


class LossDiscreteCurriculum(HeuristicDiscreteCurriculum):
    """Loss-based pacing ablation (SPL-style, untrained model forward pass)."""

    METHOD_ID = "loss_discrete"

    def _compute_difficulty(self) -> np.ndarray:
        raise RuntimeError("loss_discrete computes difficulty inside fit().")

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
        """Score losses from a single forward pass, then run the phased loop."""
        self._y_build = np.asarray(y)
        self._X_build = X
        self._texts = X_text

        t0_signals = time.perf_counter()
        self.model_ = self._init_model()
        d = sample_ce_loss(self.model_, X, y, X_text=X_text)
        r = np.zeros_like(d)
        self._coverage_score = self._coverage_score_from_signals(r, d)
        signal_time = time.perf_counter() - t0_signals

        t0_phases = time.perf_counter()
        self.phases_ = self._build_phases(r, d)
        phases_time = time.perf_counter() - t0_phases

        if recorder is not None:
            recorder.log_timing("cl_signal_extract", signal_time)
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
