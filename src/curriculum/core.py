"""Orquestrador compartilhado para estrategias de curriculum learning.

Centraliza extracao de sinais BIOIS, loop de treino faseado e registro de
metricas. Subclasses implementam apenas `_build_phases(r, e)` (ou sobrescrevem
`fit` quando precisam de feedback de loss por amostra).
"""
from __future__ import annotations

import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy import stats

from curriculum.base import CurriculumBase
from curriculum.models import CurriculumModel, LogisticRegressionModel
from results.metrics import build_phase_metrics_row

if TYPE_CHECKING:
    from results.run import RunRecorder


class BIOISCurriculumBase(CurriculumBase):
    """Base comum a todos os metodos de curriculum guiados por BIOIS."""

    def __init__(
        self,
        model: Optional[CurriculumModel] = None,
        beta: float = 0.5,
        hard_slice_quantile: float = 0.8,
        random_state: int = 42,
    ):
        self.model = model
        self.beta = beta
        self.hard_slice_quantile = hard_slice_quantile
        self.random_state = random_state

    def _extract_signals(self, selector, y):
        """Deriva (r_i, e_i) normalizados a partir de um BIOIS ja ajustado."""
        if not hasattr(selector, "_probaEveryone"):
            raise ValueError(
                "selector nao possui _probaEveryone. Garanta que BIOIS.fit "
                "foi chamado antes de instanciar o curriculum."
            )

        probas = selector._probaEveryone
        y_proba_pred = selector._y_proba_of_pred
        pred = selector._pred

        e = np.array([stats.entropy(p) for p in probas], dtype=np.float64)
        e_range = e.max() - e.min()
        if e_range > 0:
            e = (e - e.min()) / e_range
        else:
            e = np.zeros_like(e)

        r = np.array(y_proba_pred, dtype=np.float64, copy=True)
        r[pred != y] = 0.0
        r_range = r.max() - r.min()
        if r_range > 0:
            r = (r - r.min()) / r_range
        else:
            r = np.zeros_like(r)

        return r, e

    @abstractmethod
    def _build_phases(self, r, e):
        """Constroi lista de fases: [{name, indices, weights}, ...]."""
        ...

    def _init_model(self) -> CurriculumModel:
        return (
            self.model
            if self.model is not None
            else LogisticRegressionModel(random_state=self.random_state)
        )

    def _run_phase_loop(
        self,
        phases: list[dict],
        X,
        y,
        *,
        X_test=None,
        y_test=None,
        X_val=None,
        y_val=None,
        X_text=None,
        X_val_text=None,
        X_test_text=None,
        recorder: Optional["RunRecorder"] = None,
    ) -> list[dict]:
        """Executa treino e avaliacao para cada fase do schedule."""
        self.model_ = self._init_model()
        has_set_phase = hasattr(self.model_, "set_phase")
        use_text = X_text is not None

        history: list[dict] = []
        t0_cl = time.perf_counter()
        metric_eval_total = 0.0

        for phase in phases:
            indices = phase["indices"]
            weights = phase["weights"]

            if has_set_phase:
                self.model_.set_phase(phase["name"])

            if use_text:
                X_phase = [X_text[i] for i in indices]
            else:
                X_phase = X[indices]
            y_phase = y[indices]

            t0_train = time.perf_counter()
            X_phase_val = X_val_text if use_text else X_val
            self.model_.fit_stage(
                X_phase,
                y_phase,
                sample_weight=weights,
                X_val=X_phase_val,
                y_val=y_val,
            )
            train_time = time.perf_counter() - t0_train

            row = {
                "phase": phase["name"],
                "n_samples": int(len(indices)),
                "n_iter": self.model_.n_iter,
                "train_time_s": float(train_time),
                "pred_time_s": float("nan"),
                "micro_f1": float("nan"),
                "macro_f1": float("nan"),
                "f1_weighted": float("nan"),
                "accuracy": float("nan"),
                "hard_slice_quantile": float(self.hard_slice_quantile),
                "hard_slice_macro_f1": float("nan"),
                "avg_seq_len": float("nan"),
                "compute_proxy": float("nan"),
                "best_val_macro_f1": float("nan"),
                "best_val_epoch": float("nan"),
                "steps_to_best_val": float("nan"),
            }

            if X_test is not None and y_test is not None:
                X_eval = X_test_text if use_text else X_test

                t0_pred = time.perf_counter()
                proba = self.model_.predict_proba(X_eval)
                preds = np.argmax(proba, axis=1)
                pred_time = time.perf_counter() - t0_pred
                metric_eval_total += float(pred_time)

                training_stats = {}
                if hasattr(self.model_, "get_training_stats"):
                    training_stats = self.model_.get_training_stats()
                row = build_phase_metrics_row(
                    phase=phase["name"],
                    y_true=y_test,
                    y_pred=preds,
                    proba=proba,
                    n_iter=self.model_.n_iter,
                    train_time_s=train_time,
                    pred_time_s=pred_time,
                    hard_slice_quantile=self.hard_slice_quantile,
                    training_stats=training_stats,
                )

            history.append(row)
            if recorder is not None:
                recorder.log_phase(row)

        cl_total = time.perf_counter() - t0_cl
        if recorder is not None:
            recorder.log_timing("model_train_time_s", cl_total)
            recorder.log_timing("metric_eval_time_s", metric_eval_total)

        return history

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
        """Executa o curriculum: sinais -> schedule -> treino faseado."""
        t0_signals = time.perf_counter()
        r, e = self._extract_signals(selector, y)
        signal_time = time.perf_counter() - t0_signals

        t0_phases = time.perf_counter()
        self.phases_ = self._build_phases(r, e)
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
