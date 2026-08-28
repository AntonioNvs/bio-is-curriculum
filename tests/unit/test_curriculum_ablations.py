"""Unit tests for curriculum signal ablation methods."""

import numpy as np
import pytest
from scipy import sparse

from bio_is_curriculum.curriculum.methods.biois_discrete import BIOISDiscreteCurriculum
from bio_is_curriculum.curriculum.methods.heuristic_discrete import (
    LengthDiscreteCurriculum,
    LossDiscreteCurriculum,
    TfidfDiscreteCurriculum,
)
from bio_is_curriculum.curriculum.methods.registry import REGISTRY, resolve_method_id


@pytest.mark.parametrize(
    "method_id,cls",
    [
        ("length_discrete", LengthDiscreteCurriculum),
        ("loss_discrete", LossDiscreteCurriculum),
        ("tfidf_discrete", TfidfDiscreteCurriculum),
    ],
)
def test_ablation_methods_registered(method_id, cls):
    assert REGISTRY[method_id] is cls
    assert resolve_method_id(method_id) == method_id


def test_heuristic_methods_skip_biois():
    assert LengthDiscreteCurriculum.REQUIRES_BIOIS is False
    assert LossDiscreteCurriculum.REQUIRES_BIOIS is False
    assert TfidfDiscreteCurriculum.REQUIRES_BIOIS is False
    assert BIOISDiscreteCurriculum.REQUIRES_BIOIS is True


def test_length_discrete_phases_cumulative():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    texts = ["a", "a b", "a b c", "a b c d", "x", "x y", "x y z", "x y z w"]
    cur = LengthDiscreteCurriculum(q_low=0.25, q_mid=0.5, q_high=1.0)
    cur._y_build = y
    cur._texts = texts
    r, e = cur._extract_signals(None, y)
    assert np.allclose(r, 0.0)
    phases = cur._build_phases(r, e)
    assert len(phases) == 3
    n0, n1, n2 = (len(p["indices"]) for p in phases)
    assert n0 <= n1 <= n2 == len(y)
    hard_weights = phases[2]["weights"]
    assert np.allclose(hard_weights, 1.0)


def test_tfidf_discrete_phases_cumulative():
    y = np.array([0, 0, 0, 0])
    X = sparse.csr_matrix([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 1.0]])
    cur = TfidfDiscreteCurriculum(q_low=0.25, q_mid=0.5, q_high=1.0)
    cur._y_build = y
    cur._X_build = X
    r, e = cur._extract_signals(None, y)
    phases = cur._build_phases(r, e)
    assert len(phases[0]["indices"]) <= len(phases[1]["indices"]) <= len(phases[2]["indices"])


class _LossModel:
    def predict_proba(self, X):
        n = len(X)
        p = np.full((n, 2), 0.5)
        p[np.arange(n), 0] = 0.9
        return p


def test_loss_discrete_fit_runs_phase_loop(monkeypatch):
    y = np.array([0, 1, 0, 1])
    texts = ["a", "b", "c", "d"]
    cur = LossDiscreteCurriculum(q_low=0.5, q_mid=0.75, q_high=1.0, random_state=0)
    cur.model = _LossModel()

    history_rows = []

    def fake_run_phase_loop(phases, X, y, **kwargs):
        history_rows.append({"n_phases": len(phases)})
        return [{"macro_f1": 0.5}]

    monkeypatch.setattr(cur, "_run_phase_loop", fake_run_phase_loop)
    cur.fit(None, X=np.arange(4), y=y, X_text=texts)
    assert len(cur.phases_) == 3
    assert history_rows[0]["n_phases"] == 3
