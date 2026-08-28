"""Unit tests for Bengio 2009 baseline (b1)."""

import numpy as np
from sklearn.datasets import make_classification

from bio_is_curriculum.baselines.b1_bengio2009 import Baseline1


def test_b1_requires_no_biois():
    assert Baseline1.REQUIRES_BIOIS is False


def test_b1_two_phases_cumulative():
    X, y = make_classification(
        n_samples=60, n_features=16, n_informative=8,
        n_classes=3, random_state=1,
    )
    cur = Baseline1(easy_fraction=0.5, use_global_quantile=True, random_state=0)
    cur.fit(None, X, y)

    assert len(cur.phases_) == 2
    assert cur.phases_[0]["name"] == "easy"
    assert cur.phases_[1]["name"] == "target"

    easy_idx = set(cur.phases_[0]["indices"].tolist())
    target_idx = set(cur.phases_[1]["indices"].tolist())
    assert easy_idx <= target_idx
    assert len(target_idx) == len(y)
    assert len(easy_idx) == 30


def test_b1_uniform_weights():
    X, y = make_classification(
        n_samples=40, n_features=12, n_informative=6,
        n_classes=2, random_state=2,
    )
    cur = Baseline1(easy_fraction=0.5, random_state=0)
    cur.fit(None, X, y)
    for phase in cur.phases_:
        assert np.all(phase["weights"] == 1.0)
