"""Unit tests for training-dynamics difficulty signals."""

import numpy as np
import pytest

from bio_is_curriculum.signals.training_dynamics import (
    confidence_difficulty,
    gold_label_probas,
    variability_difficulty,
)


def test_gold_label_probas_direct_columns():
    probas = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    y = np.array([0, 1, 0])
    p = gold_label_probas(probas, y)
    assert p.shape == (3,)
    assert p[0] == pytest.approx(0.9)
    assert p[1] == pytest.approx(0.8)


def test_confidence_difficulty_low_confidence_is_harder():
    trace = np.array([
        [0.9, 0.95],
        [0.1, 0.2],
        [0.5, 0.5],
    ])
    d = confidence_difficulty(trace)
    assert d.shape == (3,)
    assert d[1] > d[0]
    assert d[0] < d[2]


def test_confidence_difficulty_constant_trace():
    trace = np.array([[0.5, 0.5], [0.5, 0.5]])
    d = confidence_difficulty(trace)
    assert np.allclose(d, 0.0)


def test_variability_difficulty_high_var_is_harder():
    trace = np.array([
        [0.5, 0.5],
        [0.1, 0.9],
    ])
    d = variability_difficulty(trace)
    assert d[1] > d[0]
