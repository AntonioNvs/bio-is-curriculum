"""Curriculum discreto BIOIS: fases Clean -> Diverse -> Hard."""
from __future__ import annotations

import numpy as np

from curriculum.core import BIOISCurriculumBase


class BIOISDiscreteCurriculum(BIOISCurriculumBase):
    """Curriculum learning sobre os sinais do biO-IS em tres fases discretas.

    Parameters
    ----------
    model : CurriculumModel, optional
        Modelo a ser treinado de forma faseada.
    beta : float, default=0.5
        Coeficiente de ponderacao na Fase Hard: w_i = 1 - beta * r_i.
    q_low, q_mid, q_high : float
        Quantis de entropia que delimitam as fases A, B e C.
    hard_slice_quantile : float, default=0.8
        Quantil usado para metricas de recorte dificil.
    random_state : int, default=42
        Semente usada pelo modelo default.
    """

    PHASE_NAMES = ("clean", "diverse", "hard")
    METHOD_ID = "biois_discrete"

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
    ):
        super().__init__(
            model=model,
            beta=beta,
            hard_slice_quantile=hard_slice_quantile,
            random_state=random_state,
        )
        self.q_low = q_low
        self.q_mid = q_mid
        self.q_high = q_high
        self.r_cap = r_cap

    def _build_phases(self, r, e):
        """Constroi os indices e pesos cumulativos para clean, diverse e hard."""
        n = len(e)
        e_low = np.quantile(e, self.q_low)
        e_mid = np.quantile(e, self.q_mid)
        e_high = np.quantile(e, self.q_high)

        idx_all = np.arange(n)

        mask_a = e <= e_low
        mask_b = e <= e_mid
        mask_c = e <= e_high

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
