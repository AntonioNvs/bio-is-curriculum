"""Unit tests for preprocessing helpers."""

import numpy as np

from bio_is_curriculum.data.preprocessing import subsample_train_fraction


def test_subsample_train_fraction_keeps_all_when_one():
    X = np.arange(20).reshape(-1, 1)
    y = np.array([0] * 10 + [1] * 10)
    out = subsample_train_fraction(X, y, None, None, fraction=1.0, random_state=42)
    assert out["n_before"] == 20
    assert out["n_after"] == 20
    assert len(out["y_train"]) == 20


def test_subsample_train_fraction_reduces_size():
    X = np.arange(100).reshape(-1, 1)
    y = np.array([0] * 50 + [1] * 50)
    out = subsample_train_fraction(X, y, None, None, fraction=0.2, random_state=42)
    assert out["n_before"] == 100
    assert out["n_after"] == 20
    assert len(np.unique(out["y_train"])) == 2
