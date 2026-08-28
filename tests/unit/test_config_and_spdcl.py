import numpy as np

from bio_is_curriculum.config.schema import ExperimentConfig
from bio_is_curriculum.config.loader import merge_yaml_to_experiment_config
from bio_is_curriculum.signals.nuclear_norm import (
    NuclearNormScorer,
    nuclear_norm,
    spdcl_epoch_difficulty,
)
from bio_is_curriculum.training.dynamic import progressive_bin_indices, scatter_into_bins


def test_experiment_config_roundtrip():
    cfg = ExperimentConfig(dataset="webkb", fold=1, curriculum_q=(0.2, 0.5, 0.9))
    restored = ExperimentConfig.from_dict(cfg.to_dict())
    assert restored.dataset == "webkb"
    assert restored.curriculum_q == (0.2, 0.5, 0.9)


def test_yaml_merge():
    cfg = merge_yaml_to_experiment_config({
        "dataset": "webkb",
        "instance_selection": {"beta": 0.4},
        "training": {"epochs": 3},
    })
    assert cfg.beta == 0.4
    assert cfg.epochs == 3


def test_nuclear_norm_positive():
    m = np.random.randn(8, 16)
    assert nuclear_norm(m) > 0


def test_spdcl_difficulty_first_epoch():
    cur = np.array([1.0, 2.0, 3.0])
    d = spdcl_epoch_difficulty(cur, None, first_epoch=True)
    np.testing.assert_array_equal(d, cur)


def test_spdcl_difficulty_delta():
    cur = np.array([3.0, 2.0, 1.0])
    prev = np.array([1.0, 1.0, 1.0])
    d = spdcl_epoch_difficulty(cur, prev, first_epoch=False)
    np.testing.assert_array_equal(d, np.array([2.0, 1.0, 0.0]))


def test_progressive_bins():
    bins = [np.array([0, 3]), np.array([1, 4]), np.array([2, 5])]
    idx0 = progressive_bin_indices(bins, 0)
    idx2 = progressive_bin_indices(bins, 2)
    assert set(idx0.tolist()) == {0, 3}
    assert set(idx2.tolist()) == {0, 1, 2, 3, 4, 5}


def test_scatter_into_bins():
    sorted_idx = np.arange(6)
    bins = scatter_into_bins(sorted_idx, 3)
    assert len(bins) == 3
    assert sum(len(b) for b in bins) == 6


def test_nuclear_norm_scorer_class():
    scorer = NuclearNormScorer()
    hidden = [np.random.randn(5, 16), np.random.randn(7, 16)]
    norms = scorer.score_pretrain(hidden)
    assert norms.shape == (2,)
