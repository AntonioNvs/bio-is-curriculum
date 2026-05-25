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

from src.baselines.base import BaselineBase


class Baseline1(BaselineBase):
    INDEX = 1
    NAME = "Confidence-paced CL (Bengio 2009)"
    REFERENCE = (
        "Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). "
        "Curriculum Learning. ICML 2009. "
        "https://doi.org/10.1145/1553374.1553380"
    )

    PHASE_NAMES = ("easy", "easy_medium", "all")

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

    def _build_phases(self, conf, _unused=None):
        """Pacing cumulativo por quantis de confianca (decrescente).

        Fase 1: top `q_low` mais faceis.
        Fase 2: top `q_mid` (inclui a fase 1).
        Fase 3: 100% das instancias.
        Todos os pesos sao 1.0 (Bengio CL nao pondera).
        """
        n = len(conf)
        n_easy = max(1, int(np.floor(n * self.q_low)))
        n_med = max(n_easy, int(np.floor(n * self.q_mid)))

        order = np.argsort(-conf, kind="stable")  # desc por confianca

        phases = []
        for name, k in zip(self.PHASE_NAMES, (n_easy, n_med, n)):
            indices = np.sort(order[:k])  # mantem ordem natural dos dados no batch
            weights = np.ones(len(indices), dtype=np.float64)
            phases.append({"name": name, "indices": indices, "weights": weights})
        return phases
