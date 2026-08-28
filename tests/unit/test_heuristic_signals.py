"""Unit tests for heuristic difficulty signals."""

import numpy as np
import pytest
from scipy import sparse

from bio_is_curriculum.signals.heuristics import length_difficulty
from bio_is_curriculum.signals.lexical import tfidf_difficulty
from bio_is_curriculum.signals.loss import sample_ce_loss


def test_length_difficulty_normalized():
    texts = ["a", "a b", "a b c d"]
    d = length_difficulty(texts)
    assert d.shape == (3,)
    assert d[0] == pytest.approx(0.0)
    assert d[-1] == pytest.approx(1.0)


def test_length_difficulty_constant():
    texts = ["same words", "same words"]
    d = length_difficulty(texts)
    assert np.allclose(d, 0.0)


def test_tfidf_difficulty_normalized():
    X = sparse.csr_matrix([[1.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    d = tfidf_difficulty(X)
    assert d.shape == (3,)
    assert d.min() == pytest.approx(0.0)
    assert d.max() == pytest.approx(1.0)


def test_tfidf_difficulty_constant():
    X = sparse.csr_matrix([[2.0, 0.0], [2.0, 0.0]])
    d = tfidf_difficulty(X)
    assert np.allclose(d, 0.0)


class _MockModel:
    def predict_proba(self, X):
        n = len(X)
        probas = np.full((n, 3), 1e-6)
        probas[:, 0] = 0.9
        return probas


def test_sample_ce_loss():
    model = _MockModel()
    y = np.array([0, 1, 2])
    losses = sample_ce_loss(model, X=np.arange(3), y=y)
    assert losses.shape == (3,)
    assert losses[0] < losses[1]
    assert np.all(losses > 0)
