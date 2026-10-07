"""Loss helpers para cenarios de desbalanceamento de classes.

Suporta estrategias de reweighting para classificacao single-label:
- none
- inverse_freq_ce
- effective_num_cb
- distribution_balanced (adaptacao pragmatica para single-label)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


IMBALANCE_METHODS = (
    "none",
    "inverse_freq_ce",
    "effective_num_cb",
    "distribution_balanced",
    "minority_eda",
)

LOSS_METHODS = frozenset({
    "none",
    "inverse_freq_ce",
    "effective_num_cb",
    "distribution_balanced",
})


@dataclass(frozen=True)
class ImbalanceLossConfig:
    method: str = "inverse_freq_ce"
    effective_num_beta: float = 0.9999
    dist_bal_tau: float = 1.0
    dist_bal_logit_bias: float = 0.1


@dataclass(frozen=True)
class ClassStats:
    counts: np.ndarray
    priors: np.ndarray
    n_samples: int
    num_labels: int


def validate_imbalance_method(method: str) -> str:
    key = method.strip().lower()
    if key not in IMBALANCE_METHODS:
        raise ValueError(
            f"imbalance_method {method!r} invalido. "
            f"Opcoes: {list(IMBALANCE_METHODS)}"
        )
    return key


def compute_class_stats(y: np.ndarray, num_labels: int) -> ClassStats:
    y = np.asarray(y, dtype=np.int64)
    n_samples = int(len(y))
    if n_samples == 0:
        raise ValueError("y vazio para computar estatisticas de classe.")
    counts = np.bincount(y, minlength=num_labels).astype(np.float64)
    priors = counts / max(float(n_samples), 1.0)
    return ClassStats(
        counts=counts,
        priors=priors,
        n_samples=n_samples,
        num_labels=int(num_labels),
    )


def _inverse_freq_weights(stats: ClassStats) -> np.ndarray:
    inv_freq = stats.n_samples / np.maximum(stats.counts * stats.num_labels, 1.0)
    return inv_freq


def _effective_number_weights(stats: ClassStats, beta: float) -> np.ndarray:
    beta = float(beta)
    if not (0.0 <= beta < 1.0):
        raise ValueError("effective_num_beta deve estar em [0, 1).")
    effective_num = 1.0 - np.power(beta, stats.counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    # Normaliza para media ~1 (estabilidade de gradiente)
    weights *= stats.num_labels / np.maximum(weights.sum(), 1e-12)
    return weights


def _distribution_balanced_weights(stats: ClassStats, tau: float) -> np.ndarray:
    # Adaptacao single-label:
    # - Rebalanceamento por prior (tail recebe maior peso).
    # - Exponente tau controla agressividade.
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("dist_bal_tau deve ser > 0.")
    weights = np.power(np.maximum(stats.priors, 1e-12), -tau)
    weights *= stats.num_labels / np.maximum(weights.sum(), 1e-12)
    return weights


def class_weights_from_config(
    y: np.ndarray,
    num_labels: int,
    cfg: ImbalanceLossConfig,
) -> np.ndarray | None:
    method = validate_imbalance_method(cfg.method)
    if method in {"none", "minority_eda"}:
        return None

    stats = compute_class_stats(y, num_labels)
    if method == "inverse_freq_ce":
        return _inverse_freq_weights(stats)
    if method == "effective_num_cb":
        return _effective_number_weights(stats, beta=cfg.effective_num_beta)
    if method == "distribution_balanced":
        return _distribution_balanced_weights(stats, tau=cfg.dist_bal_tau)
    raise ValueError(f"Metodo de loss nao suportado: {method}")


def maybe_logit_adjustment(
    y: np.ndarray,
    num_labels: int,
    cfg: ImbalanceLossConfig,
    device: torch.device,
) -> torch.Tensor | None:
    method = validate_imbalance_method(cfg.method)
    if method != "distribution_balanced":
        return None
    stats = compute_class_stats(y, num_labels)
    # Bias positivo para classes raras:
    # b_k = alpha * (-log p_k), com p_k = prior de classe.
    alpha = float(cfg.dist_bal_logit_bias)
    if alpha <= 0.0:
        return None
    bias = alpha * (-np.log(np.maximum(stats.priors, 1e-12)))
    return torch.tensor(bias, dtype=torch.float32, device=device)


def compute_loss_per_sample(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
    logit_adjustment: torch.Tensor | None,
) -> torch.Tensor:
    adjusted_logits = logits if logit_adjustment is None else logits + logit_adjustment
    return F.cross_entropy(
        adjusted_logits,
        labels,
        reduction="none",
        weight=class_weights,
    )
