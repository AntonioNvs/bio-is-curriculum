"""Curriculum discreto BIOIS: fases Clean -> Diverse -> Hard."""
from __future__ import annotations

import numpy as np

from curriculum.class_balance import per_class_low_quantile_mask
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
        self._y_build: np.ndarray | None = None

    def _build_phases(self, r, e):
        """Constroi os indices e pesos cumulativos para clean, diverse e hard."""
        y = self._y_build
        if y is None:
            raise RuntimeError("_y_build nao definido; chame fit() via BIOISCurriculumBase.")

        idx_all = np.arange(len(e))
        masks = (
            per_class_low_quantile_mask(e, y, self.q_low),
            per_class_low_quantile_mask(e, y, self.q_mid),
            per_class_low_quantile_mask(e, y, self.q_high),
        )

        e_mid_per_idx = np.zeros(len(e), dtype=np.float64)
        e_high_per_idx = np.zeros(len(e), dtype=np.float64)
        for cls in np.unique(y):
            cls_idx = np.flatnonzero(y == cls)
            e_mid_per_idx[cls_idx] = np.quantile(e[cls_idx], self.q_mid)
            e_high_per_idx[cls_idx] = np.quantile(e[cls_idx], self.q_high)

        phases = []
        for name, mask in zip(self.PHASE_NAMES, masks):
            indices = idx_all[mask]
            weights = np.ones(len(indices), dtype=np.float64)
            if name == "hard":
                hard_local = (e[indices] > e_mid_per_idx[indices]) & (
                    e[indices] <= e_high_per_idx[indices]
                )
                weights[hard_local] = 1.0 - self.beta * r[indices][hard_local]
                weights = np.clip(weights, 1e-6, None)
            phases.append({"name": name, "indices": indices, "weights": weights})

        return phases

    def fit(self, selector, X, y, **kwargs):
        self._y_build = np.asarray(y)
        return super().fit(selector, X, y, **kwargs)
