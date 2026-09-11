"""CL-LRC composite difficulty signal (Ranaldi et al., RANLP 2023).

Combines normalized length, unigram rarity, and Flesch-Kincaid readability.
Higher score = harder example.
"""

from __future__ import annotations

import re

import numpy as np

from bio_is_curriculum.signals.heuristics import length_difficulty

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+")


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _count_sentences(text: str) -> int:
    """Count non-empty sentence fragments split on ., !, ? (min 1 if text has words)."""
    if not text or not text.strip():
        return 0
    parts = _SENTENCE_BOUNDARY_RE.split(text)
    n = sum(1 for part in parts if part.strip())
    return max(1, n)


def _syllable_count(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def length_component(texts: list[str]) -> np.ndarray:
    """Normalized word-count difficulty (Ranaldi et al. §3.2.1)."""
    return length_difficulty(texts)


def rarity_component(texts: list[str]) -> np.ndarray:
    """Normalized unigram log-rarity difficulty (Ranaldi et al. §3.2.2)."""
    tokenized = [_tokenize(t) for t in texts]
    total_tokens = sum(len(tokens) for tokens in tokenized)
    if total_tokens == 0:
        return np.zeros(len(texts), dtype=np.float64)

    counts: dict[str, int] = {}
    for tokens in tokenized:
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1

    raw = np.zeros(len(texts), dtype=np.float64)
    for i, tokens in enumerate(tokenized):
        if not tokens:
            continue
        score = 0.0
        for tok in tokens:
            p = counts[tok] / total_tokens
            score -= float(np.log(max(p, 1e-12)))
        raw[i] = score
    return _min_max_normalize(raw)


def comprehensibility_component(texts: list[str]) -> np.ndarray:
    """Normalized Flesch-Kincaid grade level (Ranaldi et al. §3.2.3).

    Uses the standard grade-level formula with per-document sentence counting.
    Ranaldi's displayed ``/100`` scaling is an affine rescale before min-max
    normalization and does not change the ranking produced here.
    """
    raw = np.zeros(len(texts), dtype=np.float64)
    for i, text in enumerate(texts):
        tokens = _tokenize(text)
        n_words = len(tokens)
        if n_words == 0:
            continue
        n_sentences = _count_sentences(text)
        if n_sentences == 0:
            continue
        n_syllables = sum(_syllable_count(tok) for tok in tokens)
        avg_sentence_len = n_words / n_sentences
        avg_syllables_per_word = n_syllables / n_words
        raw[i] = (
            0.39 * avg_sentence_len
            + 11.8 * avg_syllables_per_word
            - 15.59
        )
    return _min_max_normalize(raw)


def lrc_difficulty(texts: list[str]) -> np.ndarray:
    """Composite LRC difficulty: dLRC = dL + dR + dC (Eq. 9)."""
    d_l = length_component(texts)
    d_r = rarity_component(texts)
    d_c = comprehensibility_component(texts)
    return d_l + d_r + d_c
