"""SPCL canonico: Self-Paced Curriculum Learning baseado em loss por amostra.

Implementacao inspirada em Kumar et al. (2010) e Jiang et al. (2014):
a cada passo, exemplos com loss abaixo do limiar lambda sao incluidos
(ponderados por max(0, lambda - loss_i)); lambda cresce ao longo do
treinamento ate cobrir todo o conjunto.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from curriculum.core import BIOISCurriculumBase

if TYPE_CHECKING:
    from results.run import RunRecorder


class SPCLLossCurriculum(BIOISCurriculumBase):
    """Self-Paced CL com feedback iterativo de loss por amostra.

    Parameters
    ----------
    model : CurriculumModel, optional
        Modelo a ser treinado dinamicamente.
    n_steps : int, default=10
        Numero de passos de pacing (aumento de lambda).
    lambda_init : float, default=0.1
        Limiar inicial de loss para inclusao de amostras.
    lambda_mult : float, default=1.5
        Fator multiplicativo de lambda a cada passo.
    min_weight : float, default=1e-3
        Peso minimo para manter amostra no batch.
    hard_slice_quantile : float, default=0.8
        Quantil usado para metricas de recorte dificil.
    random_state : int, default=42
        Semente usada pelo modelo default.
    """

    METHOD_ID = "spcl_loss"

    def __init__(
        self,
        model=None,
        beta: float = 0.5,
        n_steps: int = 10,
        lambda_init: float = 0.1,
        lambda_mult: float = 1.5,
        min_weight: float = 1e-3,
        hard_slice_quantile: float = 0.8,
        random_state: int = 42,
    ):
        super().__init__(
            model=model,
            beta=beta,
            hard_slice_quantile=hard_slice_quantile,
            random_state=random_state,
        )
        self.n_steps = max(1, n_steps)
        self.lambda_init = lambda_init
        self.lambda_mult = lambda_mult
        self.min_weight = min_weight

    def _build_phases(self, r, e):
        """Nao usado — o schedule e construido iterativamente em fit()."""
        raise NotImplementedError("SPCLLossCurriculum usa pacing baseado em loss em fit().")

    def _model_is_fitted(self) -> bool:
        """True se o backend ja foi treinado ao menos uma vez."""
        if hasattr(self.model_, "_clf"):
            return hasattr(self.model_._clf, "coef_") and self.model_._clf.coef_ is not None
        if hasattr(self.model_, "_model"):
            return self.model_._model is not None
        return False

    @staticmethod
    def _compute_sample_losses(model, X, y, X_text=None) -> np.ndarray:
        """Cross-entropy por amostra a partir de predict_proba."""
        use_text = X_text is not None
        X_input = X_text if use_text else X
        probas = model.predict_proba(X_input)
        y_arr = np.asarray(y).astype(int)
        eps = 1e-12
        p_true = np.clip(probas[np.arange(len(y_arr)), y_arr], eps, 1.0)
        return -np.log(p_true)

    def _build_loss_phase(self, losses: np.ndarray, step: int) -> dict:
        """Uma fase com pesos w_i = max(0, lambda_t - loss_i)."""
        lam = self.lambda_init * (self.lambda_mult ** (step - 1))
        weights = np.maximum(0.0, lam - losses)
        mask = weights > self.min_weight
        idx_all = np.arange(len(losses))
        indices = idx_all[mask]
        return {
            "name": f"spcl_{step:02d}",
            "indices": indices,
            "weights": weights[mask],
            "lambda": lam,
        }

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
        """Executa SPCL com recalculo de loss a cada passo."""
        t0_signals = time.perf_counter()
        _r, e = self._extract_signals(selector, y)
        signal_time = time.perf_counter() - t0_signals

        self.model_ = self._init_model()
        self.phases_ = []
        self.history_ = []

        t0_phases = time.perf_counter()
        phases_time = 0.0
        max_lambda = self.lambda_init * (self.lambda_mult ** (self.n_steps - 1))

        if recorder is not None:
            recorder.log_timing("cl_signal_extract", signal_time)

        has_set_phase = hasattr(self.model_, "set_phase")
        use_text = X_text is not None
        t0_cl = time.perf_counter()
        metric_eval_total = 0.0

        for step in range(1, self.n_steps + 1):
            t0_build = time.perf_counter()
            if self._model_is_fitted():
                losses = self._compute_sample_losses(self.model_, X, y, X_text=X_text)
            else:
                # Antes do primeiro treino, usa entropia BIOIS como proxy de loss.
                losses = e * max_lambda
            phase = self._build_loss_phase(losses, step)
            phases_time += time.perf_counter() - t0_build
            self.phases_.append(phase)

            indices = phase["indices"]
            weights = phase["weights"]

            if len(indices) == 0:
                continue

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

            from results.metrics import build_phase_metrics_row

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

            self.history_.append(row)
            if recorder is not None:
                recorder.log_phase(row)

        if recorder is not None:
            recorder.log_timing("cl_phase_build", phases_time)
            recorder.log_timing("model_train_time_s", time.perf_counter() - t0_cl)
            recorder.log_timing("metric_eval_time_s", metric_eval_total)

        self.final_losses_ = self._compute_sample_losses(self.model_, X, y, X_text=X_text)
        return self
