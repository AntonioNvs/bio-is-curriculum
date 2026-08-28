"""Augmentacao textual leve para classes minoritarias.

Implementa uma variante simples de EDA sem dependencias externas:
- random swap
- random deletion
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from collections import Counter

import numpy as np


@dataclass(frozen=True)
class AugmentationStats:
    n_before: int
    n_after: int
    n_added: int
    target_min_count: int
    aug_ratio: float
    added_by_class: dict[int, int]


def _random_swap(tokens: list[str], rng: random.Random) -> list[str]:
    if len(tokens) < 2:
        return tokens
    i, j = rng.sample(range(len(tokens)), 2)
    out = list(tokens)
    out[i], out[j] = out[j], out[i]
    return out


def _random_delete(tokens: list[str], rng: random.Random) -> list[str]:
    if len(tokens) < 2:
        return tokens
    idx = rng.randrange(len(tokens))
    out = tokens[:idx] + tokens[idx + 1 :]
    return out if out else tokens


def _augment_once(
    text: str,
    rng: random.Random,
    random_swap_prob: float,
    random_delete_prob: float,
) -> str:
    tokens = text.split()
    if not tokens:
        return text
    out = list(tokens)
    changed = False

    if rng.random() < random_swap_prob and len(out) >= 2:
        out = _random_swap(out, rng)
        changed = True
    if rng.random() < random_delete_prob and len(out) >= 2:
        out = _random_delete(out, rng)
        changed = True

    if not changed and len(out) >= 2:
        out = _random_swap(out, rng)

    return " ".join(out)


def augment_minority_texts(
    texts: list[str],
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    target_min_count: int,
    aug_ratio: float,
    random_swap_prob: float,
    random_delete_prob: float,
    random_state: int,
) -> tuple[list[str], np.ndarray, np.ndarray, AugmentationStats]:
    y = np.asarray(y, dtype=np.int64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if len(texts) != len(y) or len(y) != len(sample_weight):
        raise ValueError("texts, y e sample_weight devem ter o mesmo tamanho.")
    if target_min_count <= 0:
        raise ValueError("target_min_count deve ser > 0.")
    if aug_ratio < 0.0:
        raise ValueError("aug_ratio deve ser >= 0.")

    rng = random.Random(int(random_state))
    counts = Counter(y.tolist())
    class_to_indices: dict[int, list[int]] = {}
    for idx, cls in enumerate(y.tolist()):
        class_to_indices.setdefault(int(cls), []).append(idx)

    out_texts = list(texts)
    out_y = y.tolist()
    out_w = sample_weight.tolist()
    added_by_class: dict[int, int] = {}

    for cls, idxs in sorted(class_to_indices.items()):
        n_cls = len(idxs)
        desired = max(target_min_count, int(np.ceil(n_cls * (1.0 + aug_ratio))))
        n_new = max(0, desired - n_cls)
        if n_new == 0:
            continue
        added_by_class[int(cls)] = n_new
        for _ in range(n_new):
            src_idx = rng.choice(idxs)
            aug_text = _augment_once(
                texts[src_idx],
                rng=rng,
                random_swap_prob=random_swap_prob,
                random_delete_prob=random_delete_prob,
            )
            out_texts.append(aug_text)
            out_y.append(int(cls))
            out_w.append(float(sample_weight[src_idx]))

    stats = AugmentationStats(
        n_before=len(y),
        n_after=len(out_y),
        n_added=len(out_y) - len(y),
        target_min_count=int(target_min_count),
        aug_ratio=float(aug_ratio),
        added_by_class=added_by_class,
    )
    return out_texts, np.asarray(out_y, dtype=np.int64), np.asarray(out_w, dtype=np.float64), stats
