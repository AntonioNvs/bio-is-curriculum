"""Augmentacao textual leve para classes minoritarias.

Implementa variacoes EDA simples e deterministicas (via RNG local) sem
dependencias externas, visando datasets de tamanho medio com cauda longa.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    return _normalize_spaces(text).split(" ")


def _random_deletion(tokens: list[str], rng: np.random.Generator, p: float = 0.15) -> list[str]:
    if len(tokens) <= 2:
        return tokens
    kept = [tok for tok in tokens if rng.random() > p]
    return kept if kept else [tokens[int(rng.integers(0, len(tokens)))]]


def _random_swap(tokens: list[str], rng: np.random.Generator) -> list[str]:
    if len(tokens) <= 1:
        return tokens
    i, j = rng.integers(0, len(tokens), size=2)
    out = list(tokens)
    out[int(i)], out[int(j)] = out[int(j)], out[int(i)]
    return out


def _random_insertion(tokens: list[str], rng: np.random.Generator) -> list[str]:
    if not tokens:
        return tokens
    out = list(tokens)
    src = out[int(rng.integers(0, len(out)))]
    pos = int(rng.integers(0, len(out) + 1))
    out.insert(pos, src)
    return out


def _eda_variant(text: str, rng: np.random.Generator) -> str:
    tokens = _tokenize(text)
    if len(tokens) <= 1:
        return text
    op = int(rng.integers(0, 3))
    if op == 0:
        aug_tokens = _random_deletion(tokens, rng)
    elif op == 1:
        aug_tokens = _random_swap(tokens, rng)
    else:
        aug_tokens = _random_insertion(tokens, rng)
    aug = " ".join(aug_tokens)
    aug = _normalize_spaces(aug)
    return aug if aug else text


def augment_minority_texts(
    *,
    texts: list[str],
    y: np.ndarray,
    min_count: int = 5,
    random_state: int | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Aumenta classes com frequencia < min_count usando EDA.

    Returns:
        texts_new: textos originais + sinteticos
        y_new: labels alinhados ao texts_new
        source_idx: indices no conjunto original que originaram cada sintetico
                    (ordem igual aos append em texts_new/y_new)
    """
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        y_arr = y_arr.ravel()
    if len(texts) != len(y_arr):
        raise ValueError(f"len(texts)={len(texts)} != len(y)={len(y_arr)}")

    counts = Counter(y_arr.tolist())
    if not counts:
        return list(texts), y_arr, np.empty(0, dtype=int)

    rng = np.random.default_rng(random_state)
    new_texts = list(texts)
    new_labels: list[int] = []
    source_idx: list[int] = []

    for cls in sorted(counts):
        cls_idx = np.flatnonzero(y_arr == cls)
        k = int(cls_idx.size)
        if k == 0 or k >= min_count:
            continue
        need = min_count - k
        chosen = rng.choice(cls_idx, size=need, replace=True)
        for src in chosen.tolist():
            base_text = texts[src]
            aug_text = _eda_variant(base_text, rng)
            if aug_text == base_text:
                # Forca uma pequena variacao para evitar duplicata literal.
                aug_text = f"{base_text} {base_text.split(' ')[-1]}".strip()
            new_texts.append(aug_text)
            new_labels.append(int(cls))
            source_idx.append(int(src))

    if not source_idx:
        return new_texts, y_arr, np.empty(0, dtype=int)

    y_new = np.concatenate([y_arr, np.asarray(new_labels, dtype=y_arr.dtype)])
    return new_texts, y_new, np.asarray(source_idx, dtype=int)
