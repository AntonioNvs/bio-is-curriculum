"""SPCL soft-paced com losses iterativas e prior BIOIS.

Implementa um regime alinhado ao SPCL (AAAI 2015): a cada passo alterna
treino do modelo e atualizacao dos pesos por amostra, combinando
auto-pacing via loss (estilo scheme linear) com prior fraco de curriculum
derivado dos sinais BIOIS.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from curriculum.core import BIOISCurriculumBase
from results.metrics import build_phase_metrics_row

if TYPE_CHECKING:
    from results.run import RunRecorder


class SPCLSoftCurriculum(BIOISCurriculumBase):
    """Curriculum continuo com soft-pacing sobre sinais BIOIS.

    Parameters
    ----------
    model : CurriculumModel, optional
        Modelo a ser treinado dinamicamente.
    beta : float, default=0.5
        Coeficiente de penalizacao por redundancia.
    n_steps : int, default=6
        Numero de passos de atualizacao do curriculo (tau de 0 a 1).
    alpha_decay : float, default=10.0
        Suavidade da inclusao de exemplos mais dificeis.
    min_active_frac : float, default=0.45
        Fracao minima de exemplos ativos no inicio do curriculum.
    max_active_frac : float, default=0.95
        Fracao maxima de exemplos ativos no final do curriculum.
    pace_power : float, default=0.75
        Controla a curvatura da expansao de exemplos ativos por passo.
    lambda_init, lambda_growth, lambda_max :
        Controle do ritmo self-paced loss-driven.
    stability_tol, saturation_patience, max_effective_steps :
        Parametros de poda/early-stop para reduzir fases redundantes.
    hard_slice_quantile : float, default=0.8
        Quantil usado para metricas de recorte dificil.
    random_state : int, default=42
        Semente usada pelo modelo default.
    """

    METHOD_ID = "spcl_soft"

    def __init__(
        self,
        model=None,
        beta: float = 0.5,
        n_steps: int = 6,
        alpha_decay: float = 10.0,
        min_active_frac: float = 0.45,
        max_active_frac: float = 0.95,
        pace_power: float = 0.75,
        lambda_init: float = 0.25,
        lambda_growth: float = 1.4,
        lambda_max: float = 1.0,
        min_weight: float = 1e-3,
        stability_tol: float = 5e-3,
        saturation_patience: int = 2,
        max_effective_steps: int = 6,
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
        self.alpha_decay = alpha_decay
        self.min_active_frac = float(np.clip(min_active_frac, 0.05, 1.0))
        self.max_active_frac = float(np.clip(max_active_frac, self.min_active_frac, 1.0))
        self.pace_power = max(0.1, float(pace_power))
        self.lambda_init = float(max(1e-6, lambda_init))
        self.lambda_growth = float(max(1.0, lambda_growth))
        self.lambda_max = float(max(self.lambda_init, lambda_max))
        self.min_weight = float(max(0.0, min_weight))
        self.stability_tol = float(max(1e-8, stability_tol))
        self.saturation_patience = int(max(1, saturation_patience))
        self.max_effective_steps = int(max(1, max_effective_steps))

    def _build_phases(self, r, e):
        """Nao usado — SPCL soft e iterativo em fit()."""
        raise NotImplementedError("SPCLSoftCurriculum usa pacing iterativo em fit().")

    def _model_is_fitted(self) -> bool:
        if hasattr(self.model_, "_clf"):
            return hasattr(self.model_._clf, "coef_") and self.model_._clf.coef_ is not None
        if hasattr(self.model_, "_model"):
            return self.model_._model is not None
        return False

    @staticmethod
    def _compute_sample_losses(model, X, y, X_text=None) -> np.ndarray:
        """Cross-entropy por amostra usando predict_proba."""
        X_input = X_text if X_text is not None else X
        probas = model.predict_proba(X_input)
        y_arr = np.asarray(y).astype(int)
        if probas.shape[1] > int(np.max(y_arr)):
            p_true = probas[np.arange(len(y_arr)), y_arr]
        elif hasattr(model, "_clf") and hasattr(model._clf, "classes_"):
            cls = np.asarray(model._clf.classes_).astype(int)
            col_by_class = {c: i for i, c in enumerate(cls.tolist())}
            cols = np.array([col_by_class.get(lbl, -1) for lbl in y_arr], dtype=int)
            p_true = np.full(len(y_arr), 1e-12, dtype=np.float64)
            valid = cols >= 0
            if np.any(valid):
                p_true[valid] = probas[np.arange(len(y_arr))[valid], cols[valid]]
        else:
            p_true = np.full(len(y_arr), 1e-12, dtype=np.float64)
        p_true = np.clip(p_true, 1e-12, 1.0)
        return -np.log(p_true)

    def _build_prior_weights(self, r: np.ndarray, e: np.ndarray, tau: float) -> np.ndarray:
        """Prior fraco de curriculum: favorece menor entropia com penalizacao BIOIS."""
        diff = np.maximum(0.0, e - tau)
        w_selection = np.exp(-self.alpha_decay * (diff ** 2))
        w_penalty = np.clip(1.0 - (self.beta * r * e), 0.0, 1.0)
        return np.clip(w_selection * w_penalty, 0.0, 1.0)

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
        """SPCL soft: ACS leve com esquema linear + prior BIOIS."""
        t0_signals = time.perf_counter()
        r, e = self._extract_signals(selector, y)
        signal_time = time.perf_counter() - t0_signals

        if recorder is not None:
            recorder.log_timing("cl_signal_extract", signal_time)

        n = len(y)
        idx_all = np.arange(n)
        phases_time = 0.0
        metric_eval_total = 0.0
        self.model_ = self._init_model()
        self.phases_ = []
        self.history_ = []

        has_set_phase = hasattr(self.model_, "set_phase")
        use_text = X_text is not None
        t0_cl = time.perf_counter()

        eff_steps = min(self.n_steps, self.max_effective_steps)
        lambda_t = self.lambda_init
        prev_indices = None
        prev_weights = None
        prev_mass = 0.0
        stagnant_steps = 0

        for step in range(1, eff_steps + 1):
            tau = step / eff_steps
            prior_weights = self._build_prior_weights(r, e, tau)

            t0_build = time.perf_counter()
            if self._model_is_fitted():
                losses = self._compute_sample_losses(self.model_, X, y, X_text=X_text)
            else:
                losses = e
            v_loss = np.clip(1.0 - (losses / max(lambda_t, 1e-12)), 0.0, 1.0)
            full_weights = np.clip(v_loss * prior_weights, 0.0, 1.0)

            active_frac = self.min_active_frac + (
                (self.max_active_frac - self.min_active_frac) * (tau ** self.pace_power)
            )
            k = max(1, int(np.ceil(active_frac * n)))
            candidate = np.flatnonzero(full_weights > self.min_weight)
            if len(candidate) == 0:
                candidate = np.argpartition(full_weights, -k)[-k:]
            if len(candidate) > k:
                top_local = np.argpartition(full_weights[candidate], -k)[-k:]
                candidate = candidate[top_local]
            selected = np.sort(candidate)
            # Mantem cobertura de classes para evitar quebra de warm start em LR.
            classes_needed = np.unique(y)
            present_classes = set(np.unique(y[selected]).tolist())
            for cls in classes_needed:
                if cls in present_classes:
                    continue
                cls_idx = np.flatnonzero(y == cls)
                if cls_idx.size == 0:
                    continue
                best = cls_idx[np.argmax(full_weights[cls_idx])]
                selected = np.sort(np.append(selected, int(best)))
                present_classes.add(int(cls))

            if np.unique(y[selected]).size < 2:
                lambda_t = min(self.lambda_max, lambda_t * self.lambda_growth)
                continue
            weights = full_weights[selected]
            phases_time += time.perf_counter() - t0_build

            if prev_indices is not None and np.array_equal(selected, prev_indices):
                delta = float(np.max(np.abs(weights - prev_weights)))
                if delta < self.stability_tol:
                    lambda_t = min(self.lambda_max, lambda_t * self.lambda_growth)
                    continue

            phase = {
                "name": f"spcl_soft_{step:02d}",
                "indices": idx_all[selected],
                "weights": weights,
                "lambda": float(lambda_t),
            }
            self.phases_.append(phase)
            prev_indices = selected
            prev_weights = weights

            if has_set_phase:
                self.model_.set_phase(phase["name"])

            if use_text:
                X_phase = [X_text[i] for i in selected]
            else:
                X_phase = X[selected]
            y_phase = y[selected]

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
                "n_samples": int(len(selected)),
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

            cur_mass = float(np.mean(full_weights > self.min_weight))
            if (cur_mass - prev_mass) < self.stability_tol:
                stagnant_steps += 1
            else:
                stagnant_steps = 0
            prev_mass = cur_mass
            if stagnant_steps >= self.saturation_patience:
                break

            lambda_t = min(self.lambda_max, lambda_t * self.lambda_growth)

        if recorder is not None:
            recorder.log_timing("cl_phase_build", phases_time)
            recorder.log_timing("model_train_time_s", time.perf_counter() - t0_cl)
            recorder.log_timing("metric_eval_time_s", metric_eval_total)

        self.final_losses_ = self._compute_sample_losses(self.model_, X, y, X_text=X_text)
        return self
