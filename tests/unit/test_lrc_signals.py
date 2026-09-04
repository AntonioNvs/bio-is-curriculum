"""Unit tests for CL-LRC difficulty signals."""

import numpy as np
import pytest

from bio_is_curriculum.signals.lrc import (
    comprehensibility_component,
    length_component,
    lrc_difficulty,
    rarity_component,
)


def test_length_component_normalized():
    texts = ["a", "a b", "a b c d"]
    d = length_component(texts)
    assert d.shape == (3,)
    assert d[0] == pytest.approx(0.0)
    assert d[-1] == pytest.approx(1.0)


def test_rarity_component_rare_words_score_higher():
    texts = [
        "the cat sat",
        "the cat sat on the mat",
        "xyzzy plugh qwerty",
    ]
    d = rarity_component(texts)
    assert d.shape == (3,)
    assert d[-1] > d[0]


def test_comprehensibility_component_longer_harder():
    texts = [
        "The cat sat.",
        "The extraordinary methodological investigation continued.",
    ]
    d = comprehensibility_component(texts)
    assert d.shape == (2,)
    assert d[1] >= d[0]


def test_lrc_difficulty_monotonic_on_toy_corpus():
    texts = [
        "cat",
        "the small cat sat",
        "extraordinary methodological investigations demonstrate complexity",
    ]
    d = lrc_difficulty(texts)
    assert d.shape == (3,)
    assert d[0] <= d[1] <= d[2]


def test_lrc_difficulty_constant_texts():
    texts = ["same words here", "same words here"]
    d = lrc_difficulty(texts)
    assert np.allclose(d, d[0])
