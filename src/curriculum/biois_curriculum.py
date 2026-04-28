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
from typing import Optional

import numpy as np
from scipy import stats

from sklearn.metrics import accuracy_score, f1_score

from src.curriculum.base import CurriculumBase
from src.curriculum.models import CurriculumModel, LogisticRegressionModel


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
        preds = model.predict(X_test[mask])
        return f1_score(y_test[mask], preds, average="macro")

    def fit(self, selector, X, y, X_test=None, y_test=None):
        r, e = self._extract_signals(selector, y)
        self.phases_ = self._build_phases(r, e)

        self.model_ = (
            self.model
            if self.model is not None
            else LogisticRegressionModel(random_state=self.random_state)
        )

        self.history_ = []
        for phase in self.phases_:
            indices = phase["indices"]
            weights = phase["weights"]
            X_phase = X[indices]
            y_phase = y[indices]

            t0 = time.perf_counter()
            self.model_.fit_stage(X_phase, y_phase, sample_weight=weights)
            elapsed = time.perf_counter() - t0

            row = {
                "phase": phase["name"],
                "n_samples": int(len(indices)),
                "n_iter": self.model_.n_iter,
                "elapsed_s": float(elapsed),
            }

            if X_test is not None and y_test is not None:
                preds = self.model_.predict(X_test)
                row["micro_f1"] = float(f1_score(y_test, preds, average="micro"))
                row["macro_f1"] = float(f1_score(y_test, preds, average="macro"))
                row["accuracy"] = float(accuracy_score(y_test, preds))
                row["hard_slice_macro_f1"] = float(
                    self._hard_slice_f1(self.model_, X_test, y_test)
                )

            self.history_.append(row)

        return self
