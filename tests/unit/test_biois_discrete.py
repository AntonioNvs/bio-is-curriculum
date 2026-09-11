"""Unit tests for noise-aware BIOIS discrete curriculum."""

import numpy as np
import pytest
from types import SimpleNamespace

from bio_is_curriculum.curriculum.methods.biois_discrete import BIOISDiscreteCurriculum
from bio_is_curriculum.signals.biois import extract_biois_signals, noise_scores


def _make_selector(probas, pred, y_proba_pred):
    return SimpleNamespace(
        _probaEveryone=probas,
        _pred=np.asarray(pred),
        _y_proba_of_pred=np.asarray(y_proba_pred, dtype=np.float64),
    )


def test_noise_scores_confident_mistake_is_high():
    y = np.array([0, 1, 0])
    # Sample 1: wrong with peaked distribution (low entropy).
    probas = np.array(
        [
            [0.9, 0.1],
            [0.95, 0.05],
            [0.5, 0.5],
        ]
    )
    pred = np.array([0, 0, 0])
    selector = _make_selector(probas, pred, [0.9, 0.95, 0.5])
    noise = noise_scores(selector, y)
    assert noise[0] == pytest.approx(0.0)
    assert noise[1] > noise[2]
    assert noise[1] == pytest.approx(1.0)


def test_extract_biois_signals_returns_noise():
    y = np.array([0, 1, 0, 1])
    probas = np.array(
        [
            [0.9, 0.1],
            [0.95, 0.05],
            [0.55, 0.45],
            [0.5, 0.5],
        ]
    )
    pred = np.array([0, 0, 1, 1])
    selector = _make_selector(probas, pred, [0.9, 0.95, 0.45, 0.5])
    r, e, noise = extract_biois_signals(selector, y)
    assert r.shape == e.shape == noise.shape == (4,)
    assert noise[0] == pytest.approx(0.0)
    assert noise[1] > noise[3]
    assert noise[1] > 0.0


def test_biois_discrete_defers_confident_mistake_from_clean_phase():
    n_per_class = 8
    y = np.array([0] * n_per_class + [1] * n_per_class)
    probas = np.full((len(y), 2), 0.5)
    pred = y.copy()
    # Class 0: index 0 is easy-correct; index 1 is confident mistake.
    probas[0] = [0.95, 0.05]
    probas[1] = [0.02, 0.98]
    pred[1] = 1
    for i in range(2, n_per_class):
        probas[i] = [0.55 + 0.02 * i, 0.45 - 0.02 * i]
    for i in range(n_per_class, len(y)):
        probas[i] = [0.1, 0.9]
    y_proba_pred = [probas[i, pred[i]] for i in range(len(y))]
    selector = _make_selector(probas, pred, y_proba_pred)

    cur = BIOISDiscreteCurriculum(q_low=0.25, q_mid=0.5, q_high=1.0, beta=0.5)
    cur._y_build = y
    r, e = cur._extract_signals(selector, y)
    phases = cur._build_phases(r, e)

    clean_idx = set(phases[0]["indices"].tolist())
    class0_clean = {i for i in clean_idx if y[i] == 0}
    assert 0 in class0_clean
    assert 1 not in class0_clean


def test_biois_discrete_downweights_noisy_samples():
    n_per_class = 6
    y = np.array([0] * n_per_class + [1] * n_per_class)
    probas = np.full((len(y), 2), 0.5)
    pred = y.copy()
    probas[0] = [0.95, 0.05]
    probas[n_per_class] = [0.99, 0.01]
    pred[n_per_class] = 0
    for i in range(1, n_per_class):
        probas[i] = [0.55, 0.45]
    for i in range(n_per_class + 1, len(y)):
        probas[i] = [0.45, 0.55]
    y_proba_pred = [probas[i, pred[i]] for i in range(len(y))]
    selector = _make_selector(probas, pred, y_proba_pred)

    cur = BIOISDiscreteCurriculum(q_low=0.5, q_mid=1.0, q_high=1.0, beta=0.5)
    cur._y_build = y
    r, e = cur._extract_signals(selector, y)
    phases = cur._build_phases(r, e)

    hard_phase = phases[-1]
    weights = dict(zip(hard_phase["indices"].tolist(), hard_phase["weights"]))
    assert weights[0] == pytest.approx(1.0)
    assert weights[n_per_class] == pytest.approx(0.5)


def test_biois_discrete_uncertain_mistake_less_penalized():
    y = np.array([0, 1])
    probas = np.array([[0.95, 0.05], [0.55, 0.45]])
    pred = np.array([0, 0])
    selector = _make_selector(probas, pred, [0.95, 0.55])

    cur = BIOISDiscreteCurriculum(q_low=0.5, q_mid=1.0, q_high=1.0, beta=0.5)
    cur._y_build = y
    r, e = cur._extract_signals(selector, y)
    phases = cur._build_phases(r, e)

    hard_phase = phases[-1]
    weights = dict(zip(hard_phase["indices"].tolist(), hard_phase["weights"]))
    assert weights[1] > 0.5
