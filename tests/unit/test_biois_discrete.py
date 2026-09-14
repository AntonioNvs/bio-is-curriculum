"""Unit tests for margin/compute-aware BIOIS discrete curriculum."""

import numpy as np
import pytest
from types import SimpleNamespace

from bio_is_curriculum.curriculum.methods.biois_discrete import BIOISDiscreteCurriculum
from bio_is_curriculum.signals.biois import (
    bounded_entropy,
    extract_biois_curriculum_signals,
    extract_biois_signals,
    margin_difficulty,
    noise_scores,
    per_class_rank_normalize,
)


def _make_selector(probas, pred, y_proba_pred):
    return SimpleNamespace(
        _probaEveryone=probas,
        _pred=np.asarray(pred),
        _y_proba_of_pred=np.asarray(y_proba_pred, dtype=np.float64),
    )


def test_bounded_entropy_is_normalized():
    probas = np.array([[0.9, 0.1], [0.5, 0.5]])
    ent = bounded_entropy(probas)
    assert ent.shape == (2,)
    assert 0.0 <= ent[0] <= 1.0
    assert ent[1] > ent[0]


def test_margin_difficulty_prefers_low_margin():
    y = np.array([0, 0, 0, 0])
    probas = np.array(
        [
            [0.9, 0.1],
            [0.55, 0.45],
            [0.52, 0.48],
            [0.51, 0.49],
        ]
    )
    d = margin_difficulty(probas, y)
    assert d[0] < d[3]


def test_noise_scores_confident_mistake_is_high():
    y = np.array([0, 1, 0])
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
    assert noise[1] > 0.7


def test_extract_biois_signals_true_class_redundancy():
    y = np.array([0, 0, 1, 1])
    probas = np.array(
        [
            [0.95, 0.05],
            [0.6, 0.4],
            [0.1, 0.9],
            [0.45, 0.55],
        ]
    )
    pred = np.array([0, 0, 1, 0])
    selector = _make_selector(probas, pred, [0.95, 0.6, 0.9, 0.45])
    r, e, noise = extract_biois_signals(selector, y)
    assert r[3] == pytest.approx(0.0)
    assert r[0] > r[1]


def test_length_prior_changes_schedule_ordering():
    y = np.array([0, 0, 0, 0])
    probas = np.array(
        [
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
        ]
    )
    pred = np.zeros(4, dtype=int)
    selector = _make_selector(probas, pred, [0.9, 0.9, 0.9, 0.9])
    texts_short_first = ["a", "a b c d e", "a b", "a b c"]
    r0, e0, _ = extract_biois_curriculum_signals(
        selector, y, texts_short_first, length_weight=0.0
    )
    r1, e1, _ = extract_biois_curriculum_signals(
        selector, y, texts_short_first, length_weight=0.5
    )
    assert np.allclose(r0, r1)
    assert e1[0] < e1[1]


def test_biois_discrete_defers_confident_mistake_from_clean_phase():
    n_per_class = 8
    y = np.array([0] * n_per_class + [1] * n_per_class)
    probas = np.full((len(y), 2), 0.5)
    pred = y.copy()
    probas[0] = [0.95, 0.05]
    probas[1] = [0.02, 0.98]
    pred[1] = 1
    for i in range(2, n_per_class):
        probas[i] = [0.55 + 0.02 * i, 0.45 - 0.02 * i]
    for i in range(n_per_class, len(y)):
        probas[i] = [0.1, 0.9]
    texts = ["short"] * len(y)
    texts[1] = " ".join(["word"] * 50)
    y_proba_pred = [probas[i, pred[i]] for i in range(len(y))]
    selector = _make_selector(probas, pred, y_proba_pred)

    cur = BIOISDiscreteCurriculum(q_low=0.25, q_mid=0.5, q_high=1.0, beta=0.5)
    cur._y_build = y
    cur._texts_build = texts
    r, e = cur._extract_signals(selector, y)
    phases = cur._build_phases(r, e)

    clean_idx = set(phases[0]["indices"].tolist())
    class0_clean = {i for i in clean_idx if y[i] == 0}
    assert 0 in class0_clean
    assert 1 not in class0_clean


def test_biois_discrete_downweights_noisy_samples_only_in_hard_phase():
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
    texts = ["text"] * len(y)
    y_proba_pred = [probas[i, pred[i]] for i in range(len(y))]
    selector = _make_selector(probas, pred, y_proba_pred)

    cur = BIOISDiscreteCurriculum(q_low=0.5, q_mid=1.0, q_high=1.0, beta=0.5)
    cur._y_build = y
    cur._texts_build = texts
    r, e = cur._extract_signals(selector, y)
    phases = cur._build_phases(r, e)

    hard_weights = dict(
        zip(phases[-1]["indices"].tolist(), phases[-1]["weights"].tolist())
    )
    assert hard_weights[0] == pytest.approx(1.0)
    assert hard_weights[n_per_class] < 1.0
    assert hard_weights[n_per_class] <= 0.55
    clean_indices = set(phases[0]["indices"].tolist())
    if n_per_class in clean_indices:
        clean_weights = dict(
            zip(phases[0]["indices"].tolist(), phases[0]["weights"].tolist())
        )
        assert clean_weights[n_per_class] == pytest.approx(1.0)


def test_biois_discrete_sets_phase_max_lengths():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probas = np.full((8, 2), 0.5)
    pred = y.copy()
    selector = _make_selector(probas, pred, [0.5] * 8)
    texts = ["a"] * 8

    cur = BIOISDiscreteCurriculum(
        q_low=0.5,
        q_mid=0.75,
        q_high=1.0,
        phase_max_lengths=(96, 160, 256),
    )
    cur._y_build = y
    cur._texts_build = texts
    r, e = cur._extract_signals(selector, y)
    phases = cur._build_phases(r, e)
    assert [p["max_length"] for p in phases] == [96, 160, 256]


def test_redundancy_cap_limits_hard_phase_downweight():
    from bio_is_curriculum.curriculum.methods.discrete_base import DiscreteCurriculumBase

    class _StubDiscrete(DiscreteCurriculumBase):
        METHOD_ID = "stub_discrete"
        REQUIRES_BIOIS = False

        def _extract_signals(self, selector, y):
            return np.zeros(len(y)), np.zeros(len(y))

    y = np.zeros(8, dtype=int)
    cur = _StubDiscrete(
        q_low=1.0,
        q_mid=0.5,
        q_high=1.0,
        beta=1.0,
        r_cap=0.25,
    )
    cur._y_build = y
    r = np.zeros(8)
    r[5] = 1.0
    e = np.linspace(0.2, 0.95, 8)
    phases = cur._build_phases(r, e)
    hard = phases[-1]
    weights = dict(zip(hard["indices"].tolist(), hard["weights"].tolist()))
    assert weights[5] == pytest.approx(0.75)
