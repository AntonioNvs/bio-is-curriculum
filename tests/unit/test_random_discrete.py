"""Unit tests for the random-phase discrete curriculum (difficulty-signal control)."""

import numpy as np
from types import SimpleNamespace

from bio_is_curriculum.curriculum.methods.biois_discrete import BIOISDiscreteCurriculum
from bio_is_curriculum.curriculum.methods.random_discrete import RandomDiscreteCurriculum
from bio_is_curriculum.curriculum.methods.registry import REGISTRY, resolve_method_id


def _make_selector(probas, pred, y_proba_pred):
    return SimpleNamespace(
        _probaEveryone=probas,
        _pred=np.asarray(pred),
        _y_proba_of_pred=np.asarray(y_proba_pred, dtype=np.float64),
    )


def _fixture(n_per_class=8, n_classes=3, seed=0):
    """Balanced classes, all larger than the rare-class pinning threshold."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_classes), n_per_class)
    n = len(y)
    probas = rng.random((n, n_classes))
    probas /= probas.sum(axis=1, keepdims=True)
    pred = probas.argmax(axis=1)
    texts = [" ".join(["w"] * (1 + (i % 7))) for i in range(n)]
    return y, texts, _make_selector(probas, pred, probas[np.arange(n), pred])


def _phases(cur, y, texts, selector):
    cur._y_build = y
    cur._texts_build = texts
    r, e = cur._extract_signals(selector, y)
    return cur._build_phases(r, e)


def test_random_discrete_is_registered():
    assert REGISTRY["random_discrete"] is RandomDiscreteCurriculum
    assert resolve_method_id("random_discrete") == "random_discrete"
    assert resolve_method_id("random") == "random_discrete"


def test_random_discrete_requires_biois():
    # r and the noise term are inherited from BIOIS, so the selector is needed.
    assert RandomDiscreteCurriculum.REQUIRES_BIOIS is True


def test_random_discrete_preserves_per_class_phase_quotas():
    """The invariant the Y3 control rests on: quotas stay stratified by class.

    A global uniform difficulty still yields the same per-class quota in every
    phase, because ``per_class_low_quantile_mask`` re-quantiles per class.
    """
    y, texts, selector = _fixture()
    cur = RandomDiscreteCurriculum(
        q_low=0.25, q_mid=0.5, q_high=0.9, beta=0.5, random_state=42
    )
    phases = _phases(cur, y, texts, selector)

    sizes = [len(p["indices"]) for p in phases]
    assert 0 < sizes[0] < sizes[1] < sizes[2] <= len(y)

    for phase in phases:
        counts = [int((y[phase["indices"]] == c).sum()) for c in np.unique(y)]
        assert len(set(counts)) == 1, f"class quotas differ in {phase['name']}: {counts}"


def test_random_discrete_matches_biois_quotas():
    """Same per-class phase sizes as BIOIS under the same y and quantiles."""
    y, texts, selector = _fixture()
    kwargs = dict(q_low=0.25, q_mid=0.5, q_high=0.9, beta=0.5, random_state=42)

    biois = _phases(BIOISDiscreteCurriculum(**kwargs), y, texts, selector)
    rnd = _phases(RandomDiscreteCurriculum(**kwargs), y, texts, selector)

    for b, r in zip(biois, rnd):
        assert len(b["indices"]) == len(r["indices"])


def test_random_discrete_seed_changes_phase_assignment():
    y, texts, selector = _fixture()
    kwargs = dict(q_low=0.25, q_mid=0.5, q_high=0.9, beta=0.5)

    first = _phases(RandomDiscreteCurriculum(random_state=1, **kwargs), y, texts, selector)
    second = _phases(RandomDiscreteCurriculum(random_state=2, **kwargs), y, texts, selector)

    assert set(first[0]["indices"].tolist()) != set(second[0]["indices"].tolist())


def test_random_discrete_keeps_biois_coverage_score():
    """The coverage tie-break must follow BIOIS, not the random ordering.

    ``e`` feeds both the phase masks and the class-coverage tie-break. If the
    random vector leaked into the coverage score, the control would be "random
    phases and random coverage" rather than "random phase ordering only".
    """
    y, texts, selector = _fixture()
    cur = RandomDiscreteCurriculum(q_low=0.25, q_mid=0.5, q_high=0.9, random_state=42)
    cur._y_build = y
    cur._texts_build = texts

    r, e_random = cur._extract_signals(selector, y)
    score = cur._coverage_score_from_signals(r, e_random)

    assert cur._e_biois is not None
    assert np.allclose(score, -np.asarray(cur._e_biois, dtype=np.float64))
    assert not np.allclose(score, -e_random)


def test_random_discrete_keeps_biois_redundancy_and_noise():
    """r and the noise term stay BIOIS-derived, so hard-phase weights still apply."""
    y, texts, selector = _fixture()
    cur = RandomDiscreteCurriculum(q_low=0.25, q_mid=0.5, q_high=1.0, beta=0.5, random_state=42)
    phases = _phases(cur, y, texts, selector)

    assert cur._noise_ is not None
    assert not np.allclose(phases[2]["weights"], 1.0)
    assert np.allclose(phases[0]["weights"], 1.0)
