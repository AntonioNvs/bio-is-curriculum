"""SPCL canonico (Jiang, Meng, Zhao, Shan, Hauptmann; AAAI 2015).

Implementa o Algoritmo 1 do paper "Self-Paced Curriculum Learning":

    min_{w, v in [0,1]^n}  sum_i v_i L(y_i, g(x_i, w))  +  f(v; lambda)
    sujeito a              v in Psi

onde Psi e a regiao de curriculum (constraint linear `a^T v <= c`,
Theorem 1 do paper) que codifica conhecimento previo, e f(v;lambda) e
uma self-paced function (binary/linear/log/mixture, Eqs. 4-7).

No contexto deste repositorio, o prior `a` e derivado dos sinais BIOIS:
entropia `e` e (opcionalmente) reliability `r`. Amostras com menor
entropia BIOIS recebem `a_i` menor e portanto sao priorizadas no
curriculum, consistente com Theorem 1.
"""
from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

from curriculum.core import BIOISCurriculumBase

if TYPE_CHECKING:
    from results.run import RunRecorder


_VALID_SCHEMES = ("binary", "linear", "log", "mixture")


class SPCLLossCurriculum(BIOISCurriculumBase):
    """SPCL canonico com regiao de curriculum BIOIS e schemes do paper.

    Parameters
    ----------
    model : CurriculumModel, optional
        Backend treinado dinamicamente (LR ou RoBERTa via warm-start).
    beta : float, default=0.5
        Coeficiente do prior `a_i = e_i + beta * r_i * e_i` (Theorem 1).
    scheme : {"binary", "linear", "log", "mixture"}, default="linear"
        Self-paced function f(v; lambda) usada para fechar v*.
    n_steps : int, default=10
        Numero de iteracoes ACS (linha 3 do Algorithm 1).
    lambda_init : float, default=0.5
        Lambda inicial (model "age"). Cross-entropy multiclass tipica
        e ordem 0.5-2.0 no inicio do treino.
    lambda_step : float, default=0.5
        Incremento aditivo de lambda por iteracao (mu na linha 6 do
        Algorithm 1). So e usado se `lambda_mult <= 1.0`.
    lambda_mult : float, default=1.0
        Multiplicador geometrico de lambda (alternativa ao incremento
        aditivo). Quando `> 1.0`, sobrescreve `lambda_step` e a
        atualizacao vira `lambda *= lambda_mult` (compat. com CLI antigo).
    lambda_max : float, optional
        Teto opcional para lambda. Sem teto por default.
    lambda2 : float, optional
        Segundo limiar do scheme "mixture" (lambda1 > lambda2 > 0).
        Default = lambda_init / 2.
    prior_use_reliability : bool, default=True
        Se True, usa `a = e + beta * r * e`. Se False, usa `a = e`.
    min_weight : float, default=1e-3
        Limiar para considerar amostra ativa numa fase (evita batches
        com pesos numericamente nulos).
    hard_slice_quantile : float, default=0.8
        Quantil para metricas de hard slice (passado para a base).
    random_state : int, default=42
        Semente do modelo default.
    """

    METHOD_ID = "spcl_loss"

    def __init__(
        self,
        model=None,
        beta: float = 0.5,
        scheme: str = "linear",
        n_steps: int = 10,
        lambda_init: float = 0.5,
        lambda_step: float = 0.5,
        lambda_mult: float = 1.0,
        lambda_max: Optional[float] = None,
        lambda2: Optional[float] = None,
        prior_use_reliability: bool = True,
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
        if scheme not in _VALID_SCHEMES:
            raise ValueError(
                f"scheme {scheme!r} invalido. Use um de: {_VALID_SCHEMES}"
            )
        self.scheme = scheme
        self.n_steps = max(1, int(n_steps))
        self.lambda_init = float(max(1e-6, lambda_init))
        self.lambda_step = float(max(0.0, lambda_step))
        self.lambda_mult = float(max(1.0, lambda_mult))
        self.lambda_max = None if lambda_max is None else float(lambda_max)
        self.lambda2 = None if lambda2 is None else float(lambda2)
        self.prior_use_reliability = bool(prior_use_reliability)
        self.min_weight = float(max(0.0, min_weight))

    def _build_phases(self, r, e):
        """Nao usado - schedule e construido iterativamente em fit()."""
        raise NotImplementedError(
            "SPCLLossCurriculum usa Algorithm 1 iterativo em fit()."
        )

    def _model_is_fitted(self) -> bool:
        if hasattr(self.model_, "_clf"):
            return (
                hasattr(self.model_._clf, "coef_")
                and self.model_._clf.coef_ is not None
            )
        if hasattr(self.model_, "_model"):
            return self.model_._model is not None
        return False

    @staticmethod
    def _compute_sample_losses(model, X, y, X_text=None) -> np.ndarray:
        """Cross-entropy por amostra a partir de predict_proba.

        Robusto a classes ausentes no batch da fase anterior (LR
        multiclass com warm-start pode ter `classes_` parcial).
        """
        X_input = X_text if X_text is not None else X
        probas = model.predict_proba(X_input)
        y_arr = np.asarray(y).astype(int)
        eps = 1e-12
        if probas.shape[1] > int(np.max(y_arr)):
            p_true = probas[np.arange(len(y_arr)), y_arr]
        elif hasattr(model, "_clf") and hasattr(model._clf, "classes_"):
            cls = np.asarray(model._clf.classes_).astype(int)
            col_by_class = {c: i for i, c in enumerate(cls.tolist())}
            cols = np.array(
                [col_by_class.get(int(lbl), -1) for lbl in y_arr], dtype=int
            )
            p_true = np.full(len(y_arr), eps, dtype=np.float64)
            valid = cols >= 0
            if np.any(valid):
                p_true[valid] = probas[np.arange(len(y_arr))[valid], cols[valid]]
        else:
            p_true = np.full(len(y_arr), eps, dtype=np.float64)
        p_true = np.clip(p_true, eps, 1.0)
        return -np.log(p_true)

    # ------------------------------------------------------------------
    # SPCL building blocks (paper Eqs. 4-7 e Theorem 1).
    # ------------------------------------------------------------------

    def _build_curriculum_region(self, r: np.ndarray, e: np.ndarray):
        """Constroi (a, c) tais que Psi = {v : a^T v <= c, v in [0,1]^n}.

        Theorem 1 exige a_i < a_j quando gamma(x_i) < gamma(x_j); usamos
        entropia BIOIS (com penalizacao opcional por reliability) como
        sinal de dificuldade. `c = sum(a)` faz Psi conter o hipercubo
        unitario - prior fraco que apenas re-pondera a projecao.
        """
        a = e.astype(np.float64, copy=True)
        if self.prior_use_reliability:
            a = a + self.beta * r * e
        a_max = float(np.max(a)) if a.size else 0.0
        if a_max > 0.0:
            a = a / a_max
        c = float(a.sum())
        return a, c

    def _solve_v_star(self, losses: np.ndarray, lam: float) -> np.ndarray:
        """Closed-form v* dado w fixo, por scheme (paper, Eqs. 2/5/6/7)."""
        L = np.asarray(losses, dtype=np.float64)
        if self.scheme == "binary":
            v = (L < lam).astype(np.float64)
        elif self.scheme == "linear":
            v = np.clip(1.0 - L / max(lam, 1e-12), 0.0, 1.0)
        elif self.scheme == "log":
            lam_eff = float(lam)
            if not (0.0 < lam_eff < 1.0):
                warnings.warn(
                    f"log scheme exige lambda em (0,1); clipando lambda="
                    f"{lam_eff:.4f} -> {min(0.999, max(1e-3, lam_eff)):.4f}",
                    RuntimeWarning,
                )
                lam_eff = min(0.999, max(1e-3, lam_eff))
            zeta = 1.0 - lam_eff
            v = np.zeros_like(L)
            mask = (L < lam_eff) & (L + zeta > 0.0)
            if np.any(mask):
                # log(zeta) < 0; razao log(L+zeta)/log(zeta) cresce com L
                # decrescente, ficando em [0, 1] na regiao valida.
                v[mask] = np.log(L[mask] + zeta) / np.log(zeta)
            v = np.clip(v, 0.0, 1.0)
        elif self.scheme == "mixture":
            lam1 = float(lam)
            lam2 = float(self.lambda2) if self.lambda2 is not None else lam1 / 2.0
            if not (lam1 > lam2 > 0.0):
                # Falha silenciosa graciosa: cai para linear neste step.
                warnings.warn(
                    f"mixture scheme exige lambda1>lambda2>0 (lam1={lam1}, "
                    f"lam2={lam2}); usando linear neste step.",
                    RuntimeWarning,
                )
                v = np.clip(1.0 - L / max(lam1, 1e-12), 0.0, 1.0)
            else:
                zeta = (lam1 * lam2) / (lam1 - lam2)
                v = np.zeros_like(L)
                hard_one = L <= (zeta * lam1) / (lam1 + zeta)
                hard_zero = L >= lam1
                soft = ~(hard_one | hard_zero)
                v[hard_one] = 1.0
                if np.any(soft):
                    v[soft] = np.clip(
                        zeta / np.maximum(L[soft], 1e-12) - zeta / lam1,
                        0.0,
                        1.0,
                    )
        else:  # pragma: no cover - validado em __init__
            raise ValueError(f"scheme invalido: {self.scheme}")
        return v

    @staticmethod
    def _project_onto_psi(
        v: np.ndarray, a: np.ndarray, c: float
    ) -> np.ndarray:
        """Projeta v em Psi = {a^T v <= c, v in [0,1]^n}.

        Para c = sum(a) (default), `a^T v <= sum(a)` e sempre satisfeita
        - retorna v inalterado. Caso contrario, zera amostras de menor
        razao `v_i / a_i` ate a constraint passar (greedy feasible).
        """
        ay = float(np.dot(a, v))
        if ay <= c + 1e-12:
            return v
        v = v.copy()
        eps = 1e-12
        ratio = v / (a + eps)
        # Considera apenas amostras com v_i > 0; zera as de pior razao.
        order = np.argsort(ratio)
        for idx in order:
            if v[idx] <= 0.0:
                continue
            ay -= a[idx] * v[idx]
            v[idx] = 0.0
            if ay <= c + 1e-12:
                break
        return v

    @staticmethod
    def _ensure_class_coverage(
        selected: np.ndarray,
        full_weights: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Garante que todas as classes presentes em y tenham >=1 amostra."""
        classes_needed = np.unique(y)
        present = set(np.unique(y[selected]).tolist())
        for cls in classes_needed:
            cls_int = int(cls)
            if cls_int in present:
                continue
            cls_idx = np.flatnonzero(y == cls)
            if cls_idx.size == 0:
                continue
            best = cls_idx[np.argmax(full_weights[cls_idx])]
            selected = np.sort(np.append(selected, int(best)))
            present.add(cls_int)
        return selected

    def _next_lambda(self, lam: float) -> float:
        if self.lambda_mult > 1.0:
            new_lam = lam * self.lambda_mult
        else:
            new_lam = lam + self.lambda_step
        if self.lambda_max is not None:
            new_lam = min(self.lambda_max, new_lam)
        return float(new_lam)

    # ------------------------------------------------------------------
    # fit: Algorithm 1 do paper.
    # ------------------------------------------------------------------

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
        """Executa SPCL: alterna treino do modelo e atualizacao de v em Psi."""
        t0_signals = time.perf_counter()
        r, e = self._extract_signals(selector, y)
        signal_time = time.perf_counter() - t0_signals

        # Linha 1 do Algorithm 1: derivar Psi do curriculum.
        a, c = self._build_curriculum_region(r, e)
        self.curriculum_a_ = a
        self.curriculum_c_ = c

        self.model_ = self._init_model()
        self.phases_ = []
        self.history_ = []

        if recorder is not None:
            recorder.log_timing("cl_signal_extract", signal_time)

        has_set_phase = hasattr(self.model_, "set_phase")
        use_text = X_text is not None
        idx_all = np.arange(len(y))

        # Linha 2: inicializa lambda na regiao viavel.
        lam = self.lambda_init

        phases_time = 0.0
        metric_eval_total = 0.0
        t0_cl = time.perf_counter()

        for step in range(1, self.n_steps + 1):
            # Losses para atualizar v (Step 5 do Algorithm 1).
            t0_build = time.perf_counter()
            if self._model_is_fitted():
                losses = self._compute_sample_losses(
                    self.model_, X, y, X_text=X_text
                )
            else:
                # Antes do primeiro treino, usa entropia BIOIS como proxy
                # (escalado para a faixa de lambda atual): preserva a
                # filosofia "easy first" no warm-start.
                losses = e * lam

            v = self._solve_v_star(losses, lam)
            v = self._project_onto_psi(v, a, c)
            mask = v > self.min_weight
            selected = idx_all[mask]
            selected = self._ensure_class_coverage(selected, v, y)
            phases_time += time.perf_counter() - t0_build

            if selected.size == 0 or np.unique(y[selected]).size < 2:
                # Lambda muito pequeno - cresce o limiar e tenta novamente.
                lam = self._next_lambda(lam)
                continue

            weights = v[selected]
            # Evita w=0 nos indices forcados por class-coverage.
            weights = np.where(weights > 0.0, weights, self.min_weight)

            phase = {
                "name": f"spcl_{step:02d}",
                "indices": selected,
                "weights": weights,
                "lambda": float(lam),
                "scheme": self.scheme,
            }
            self.phases_.append(phase)

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

            from results.metrics import build_phase_metrics_row

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

            lam = self._next_lambda(lam)

        if recorder is not None:
            recorder.log_timing("cl_phase_build", phases_time)
            recorder.log_timing("model_train_time_s", time.perf_counter() - t0_cl)
            recorder.log_timing("metric_eval_time_s", metric_eval_total)

        if self._model_is_fitted():
            self.final_losses_ = self._compute_sample_losses(
                self.model_, X, y, X_text=X_text
            )
        else:
            self.final_losses_ = np.full(len(y), float("nan"))
        return self
