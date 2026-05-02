"""biO-IS-Curriculum: curriculum learning guiado por sinais do biO-IS.

Implementacao minima (primeiro passo) que consome os escores de
redundancia (r_i) e entropia (e_i) ja calculados pelo `BIOIS` e organiza
o treinamento em tres fases (Clean -> Diverse -> Hard), conforme
descrito na proposta do projeto.

O modelo utilizado em cada fase e fornecido como uma abstracao
(`CurriculumModel`), de modo que esta classe nao depende de nenhuma
implementacao concreta.
"""
import time
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy import stats

from sklearn.metrics import accuracy_score, f1_score

from src.curriculum.base import CurriculumBase
from src.curriculum.models import CurriculumModel, LogisticRegressionModel

if TYPE_CHECKING:
    from src.results.run import RunRecorder


class BIOISCurriculum(CurriculumBase):
    """Curriculum learning sobre os sinais do biO-IS.

    Parameters
    ----------
    model : CurriculumModel, optional
        Modelo a ser treinado de forma faseada. Se `None`, usa
        `LogisticRegressionModel` como default.

    beta : float, default=0.5
        Coeficiente de ponderacao na Fase C: w_i = 1 - beta * r_i.

    q_low, q_mid, q_high : float
        Quantis de entropia que delimitam as fases A, B e C.

    r_cap : float, default=0.5
        Quantil de redundancia usado como teto na Fase A
        (proxy para "um representante por cluster").

    random_state : int, default=42
        Semente usada apenas pelo modelo default.

    Attributes
    ----------
    phases_ : list of dict
        Indices e pesos de cada fase (cumulativos).

    history_ : list of dict
        Metricas registradas apos o treino de cada fase.

    model_ : CurriculumModel
        Modelo final treinado (referencia ao parametro `model`, ou ao
        default criado internamente).
    """

    PHASE_NAMES = ("clean", "diverse", "hard")

    def __init__(
        self,
        model: Optional[CurriculumModel] = None,
        beta: float = 0.5,
        q_low: float = 0.3,
        q_mid: float = 0.6,
        q_high: float = 0.95,
        r_cap: float = 0.5,
        random_state: int = 42,
    ):
        self.model = model
        self.beta = beta
        self.q_low = q_low
        self.q_mid = q_mid
        self.q_high = q_high
        self.r_cap = r_cap
        self.random_state = random_state

    def _extract_signals(self, selector, y):
        """Deriva (r_i, e_i) a partir de um BIOIS ja ajustado."""
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

    def _build_phases(self, r, e):
        """Constroi os indices e pesos cumulativos para A, B e C."""
        n = len(e)
        e_low = np.quantile(e, self.q_low)
        e_mid = np.quantile(e, self.q_mid)
        e_high = np.quantile(e, self.q_high)
        r_threshold = np.quantile(r, self.r_cap)

        idx_all = np.arange(n)

        mask_a = (e <= e_low) & (r <= r_threshold)
        mask_b = mask_a | ((e > e_low) & (e <= e_mid))
        mask_c = mask_b | ((e > e_mid) & (e <= e_high))

        phases = []
        for name, mask in zip(self.PHASE_NAMES, (mask_a, mask_b, mask_c)):
            indices = idx_all[mask]
            weights = np.ones(len(indices), dtype=np.float64)
            if name == "hard":
                hard_local = (e[indices] > e_mid) & (e[indices] <= e_high)
                weights[hard_local] = 1.0 - self.beta * r[indices][hard_local]
                weights = np.clip(weights, 1e-6, None)
            phases.append({"name": name, "indices": indices, "weights": weights})

        return phases

    @staticmethod
    def _hard_slice_f1(model: CurriculumModel, X_test, y_test, top_q: float = 0.8):
        """F1-macro no subconjunto de teste com maior entropia preditiva."""
        probs = model.predict_proba(X_test)
        ent = np.array([stats.entropy(p) for p in probs])
        threshold = np.quantile(ent, top_q)
        mask = ent >= threshold
        if mask.sum() == 0:
            return float("nan")
        preds = model.predict(X_test[mask] if not isinstance(X_test, list) else [X_test[i] for i in np.where(mask)[0]])
        return f1_score(y_test[mask], preds, average="macro")

    def fit(
        self,
        selector,
        X,
        y,
        X_test=None,
        y_test=None,
        X_text=None,
        X_test_text=None,
        recorder: "Optional[RunRecorder]" = None,
    ):
        """Executa o curriculum faseado.

        Parameters
        ----------
        selector : BIOIS (ja ajustado)
        X : csr_matrix -- features TF-IDF (usadas pelo LogisticRegressionModel)
        y : ndarray -- labels de treino
        X_test, y_test : csr_matrix, ndarray -- conjunto de teste para metricas
        X_text : list[str], optional -- textos de treino (usados pelo RobertaModel)
        X_test_text : list[str], optional -- textos de teste
        recorder : RunRecorder, optional -- grava metricas em disco
        """
        t0_signals = time.perf_counter()
        r, e = self._extract_signals(selector, y)
        signal_time = time.perf_counter() - t0_signals

        t0_phases = time.perf_counter()
        self.phases_ = self._build_phases(r, e)
        phases_time = time.perf_counter() - t0_phases

        if recorder is not None:
            recorder.log_timing("cl_signal_extract", signal_time)
            recorder.log_timing("cl_phase_build", phases_time)

        self.model_ = (
            self.model
            if self.model is not None
            else LogisticRegressionModel(random_state=self.random_state)
        )

        # Informa o nome da fase ao modelo (usado para logs por RobertaModel)
        has_set_phase = hasattr(self.model_, "set_phase")

        # Decide qual representacao usar: texto (RoBERTa) ou matriz (LR)
        use_text = X_text is not None

        self.history_ = []
        t0_cl = time.perf_counter()

        for phase in self.phases_:
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
            self.model_.fit_stage(X_phase, y_phase, sample_weight=weights)
            train_time = time.perf_counter() - t0_train

            row = {
                "phase": phase["name"],
                "n_samples": int(len(indices)),
                "n_iter": self.model_.n_iter,
                "train_time_s": float(train_time),
                "pred_time_s": float("nan"),
                "micro_f1": float("nan"),
                "macro_f1": float("nan"),
                "accuracy": float("nan"),
                "hard_slice_macro_f1": float("nan"),
            }

            if X_test is not None and y_test is not None:
                X_eval = X_test_text if use_text else X_test

                t0_pred = time.perf_counter()
                preds = self.model_.predict(X_eval)
                pred_time = time.perf_counter() - t0_pred

                row["pred_time_s"] = float(pred_time)
                row["micro_f1"] = float(f1_score(y_test, preds, average="micro"))
                row["macro_f1"] = float(f1_score(y_test, preds, average="macro"))
                row["accuracy"] = float(accuracy_score(y_test, preds))
                row["hard_slice_macro_f1"] = float(
                    self._hard_slice_f1(self.model_, X_eval, y_test)
                )

            self.history_.append(row)
            if recorder is not None:
                recorder.log_phase(row)

        cl_total = time.perf_counter() - t0_cl
        if recorder is not None:
            recorder.log_timing("cl_total", cl_total)

        return self
