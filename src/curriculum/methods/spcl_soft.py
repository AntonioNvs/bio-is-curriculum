"""SPCL soft-paced: curriculum continuo sobre sinais de entropia/redundancia BIOIS.

Substitui fases discretas por ponderacao continua (soft-pacing) inspirada em
Self-Paced Curriculum Learning, usando os escores (r_i, e_i) do BIOIS como
proxy de dificuldade em vez de loss iterativo.
"""
from __future__ import annotations

import numpy as np

from curriculum.core import BIOISCurriculumBase


class SPCLSoftCurriculum(BIOISCurriculumBase):
    """Curriculum continuo com soft-pacing sobre sinais BIOIS.

    Parameters
    ----------
    model : CurriculumModel, optional
        Modelo a ser treinado dinamicamente.
    beta : float, default=0.5
        Coeficiente de penalizacao por redundancia.
    n_steps : int, default=10
        Numero de passos de atualizacao do curriculo (tau de 0 a 1).
    alpha_decay : float, default=10.0
        Suavidade da inclusao de exemplos mais dificeis.
    min_active_frac : float, default=0.45
        Fracao minima de exemplos ativos no inicio do curriculum.
    max_active_frac : float, default=0.95
        Fracao maxima de exemplos ativos no final do curriculum.
    pace_power : float, default=0.75
        Controla a curvatura da expansao de exemplos ativos por passo.
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
        n_steps: int = 10,
        alpha_decay: float = 10.0,
        min_active_frac: float = 0.45,
        max_active_frac: float = 0.95,
        pace_power: float = 0.75,
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

    def _build_phases(self, r, e):
        """Constroi passos com ponderacao continua (soft-pacing)."""
        n = len(e)
        idx_all = np.arange(n)
        phases = []
        prev_indices = None
        prev_weights = None

        for step in range(1, self.n_steps + 1):
            tau = step / self.n_steps

            diff = np.maximum(0.0, e - tau)
            w_selection = np.exp(-self.alpha_decay * (diff ** 2))
            w_penalty = 1.0 - (self.beta * r * e)
            weights = w_selection * w_penalty
            weights = np.clip(weights, 1e-6, 1.0)

            active_frac = self.min_active_frac + (
                (self.max_active_frac - self.min_active_frac) * (tau ** self.pace_power)
            )
            k = max(1, int(np.ceil(active_frac * n)))

            # Mantem somente a massa principal de exemplos por passo para reduzir
            # custo sem alterar o ranqueamento soft de dificuldade.
            selected = np.argpartition(weights, -k)[-k:]
            selected = np.sort(selected)
            indices = idx_all[selected]
            step_weights = weights[selected]

            if prev_indices is not None and np.array_equal(indices, prev_indices):
                delta = float(np.max(np.abs(step_weights - prev_weights)))
                if delta < 1e-3:
                    continue

            phases.append({
                "name": f"step_{step:02d}",
                "indices": indices,
                "weights": step_weights,
            })
            prev_indices = indices
            prev_weights = step_weights

        return phases
