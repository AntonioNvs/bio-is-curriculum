"""Unit tests for oracle-margin difficulty scoring."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer

from bio_is_curriculum.signals.oracle_margin import (
    easy_indices_by_margin,
    multiclass_margins,
    oof_lr_probas,
)


def test_multiclass_margins_higher_when_correct_and_confident():
    probas = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
        [0.5, 0.5],
    ])
    y = np.array([1, 0, 0])
    margins = multiclass_margins(probas, y)
    assert margins[0] == pytest.approx(0.8)
    assert margins[1] == pytest.approx(0.6)
    assert margins[2] == pytest.approx(0.0)


def test_oof_lr_probas_shape():
    X, y = make_classification(
        n_samples=80, n_features=20, n_informative=10,
        n_classes=3, random_state=0,
    )
    probas = oof_lr_probas(X, y, random_state=0, n_splits=4)
    assert probas.shape == (80, 3)
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)


def test_easy_indices_global_fraction():
    margin = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    y = np.array([0, 0, 1, 1, 1])
    easy = easy_indices_by_margin(margin, y, 0.4, use_global=True)
    assert len(easy) == 2
    assert set(easy.tolist()) == {0, 1}


def test_easy_indices_tfidf_smoke():
    texts = ["cat sat", "dog ran", "bird flew", "fish swam"] * 5
    y = np.array([0, 1, 0, 1] * 5)
    X = TfidfVectorizer().fit_transform(texts)
    probas = oof_lr_probas(X, y, random_state=0, n_splits=2)
    margin = multiclass_margins(probas, y)
    easy = easy_indices_by_margin(margin, y, 0.5, use_global=True)
    assert easy.size >= 1
    assert easy.size <= len(y)
