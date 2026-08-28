"""Unit tests for SPDCL schedule semantics."""

import numpy as np

from bio_is_curriculum.signals.nuclear_norm import NuclearNormScorer
from bio_is_curriculum.training.dynamic import (
    progressive_bin_indices,
    scatter_into_bins,
)


def test_nuclear_norm_scorer_pretrain_and_delta():
    hidden = [np.ones((4, 8)), np.ones((6, 8)) * 2]
    scorer = NuclearNormScorer()
    pre = scorer.score_pretrain(hidden)
    assert pre.shape == (2,)
    assert pre[1] > pre[0]

    current = np.array([3.0, 4.0])
    delta = scorer.score_delta(current)
    np.testing.assert_array_equal(delta, current - pre)


def test_curriculum_epochs_cover_all_samples():
    n = 12
    n_bins = 3
    order = np.arange(n)
    bins = scatter_into_bins(order, n_bins)
    for epoch in range(n_bins):
        idx = progressive_bin_indices(bins, epoch)
        assert len(idx) > 0
    all_visible = progressive_bin_indices(bins, n_bins - 1)
    assert set(all_visible.tolist()) == set(range(n))


def test_delta_sort_direction():
    difficulty = np.array([0.1, 0.5, 0.3, 0.9])
    order = np.argsort(-difficulty)
    assert order[0] == 3  # largest delta first
