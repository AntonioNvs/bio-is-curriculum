"""Baseline 1 — Confidence-paced Curriculum Learning (Bengio et al. 2009).

Currículo ingênuo do mais fácil para o mais difícil, onde "dificuldade"
é a confiança do classificador fraco no rótulo verdadeiro.

Reaproveita `_probaEveryone` já calculado pelo `BIOIS.fitting_alpha`
(LR multinomial em 5-fold CV sobre o TF-IDF), evitando treinar um
classificador adicional. As fases são cumulativas (top-q_low → top-q_mid
→ tudo) usando os mesmos quantis do `is_cl` para comparação justa de
schedule. Não há `sample_weight`, não há remoção de ruído, não há
peso de redundância — exatamente o que `is_cl` acrescenta a este
baseline.

Referência
----------
Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).
Curriculum Learning. ICML 2009.
https://doi.org/10.1145/1553374.1553380
"""
from __future__ import annotations

import numpy as np

from bio_is_curriculum.baselines.base import BaselineBase
from bio_is_curriculum.curriculum.class_balance import per_class_high_quantile_mask


class Baseline1(BaselineBase):
    INDEX = 1
    NAME = "Confidence-paced CL (Bengio 2009)"
    REFERENCE = (
        "Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). "
        "Curriculum Learning. ICML 2009. "
        "https://doi.org/10.1145/1553374.1553380"
    )

    PHASE_NAMES = ("easy", "easy_medium", "all")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._y_build: np.ndarray | None = None

    def _extract_signals(self, selector, y):
        """Sinal único: confiança do LR fraco no rótulo verdadeiro de cada exemplo.

        Retorna `(conf, conf)` para casar a assinatura `(r, e)` do pai —
        `_build_phases` ignora o segundo termo.
        """
        if not hasattr(selector, "_probaEveryone"):
            raise ValueError(
                "selector nao possui _probaEveryone. Garanta que BIOIS.fit "
                "foi chamado antes de instanciar o baseline."
            )
        probas = np.asarray(selector._probaEveryone)
        y_arr = np.asarray(y).astype(int)
        conf = probas[np.arange(len(y_arr)), y_arr]
        return conf, conf

    def _coverage_score_from_signals(self, r: np.ndarray, e: np.ndarray) -> np.ndarray:
        return np.asarray(e, dtype=np.float64)

    def _build_phases(self, conf, _unused=None):
        """Pacing cumulativo por quantis de confianca estratificados por classe."""
        y = self._y_build
        if y is None:
            raise RuntimeError("_y_build nao definido; chame fit() via BIOISCurriculumBase.")

        idx_all = np.arange(len(conf))
        masks = (
            per_class_high_quantile_mask(conf, y, self.q_low),
            per_class_high_quantile_mask(conf, y, self.q_mid),
            np.ones(len(conf), dtype=bool),
        )

        phases = []
        for name, mask in zip(self.PHASE_NAMES, masks):
            indices = idx_all[mask]
            weights = np.ones(len(indices), dtype=np.float64)
            phases.append({"name": name, "indices": indices, "weights": weights})
        return phases

    def fit(self, selector, X, y, **kwargs):
        self._y_build = np.asarray(y)
        return super().fit(selector, X, y, **kwargs)
