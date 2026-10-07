"""Unit tests for self-adaptive PLM scoring and PMD sampling."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bio_is_curriculum.curriculum.methods.registry import (
    REGISTRY,
    build_curriculum_kwargs,
    resolve_method_id,
)
from bio_is_curriculum.curriculum.methods.self_adaptive_pmd import (
    SelfAdaptivePMDCurriculum,
)
from bio_is_curriculum.models.modernbert import PMDBatchSampler
from bio_is_curriculum.signals.self_adaptive import (
    confidence_from_class_probs,
    difficulty_from_confidence,
    pmd_partition_counts,
    pmd_rank_weights,
    pmd_sampling_probs,
    select_verbalizers,
)


class _FakeTok:
    """Minimal tokenizer stub for verbalizer eligibility checks."""

    def __init__(self, vocab: dict[str, int]):
        self._vocab = dict(vocab)
        self._id_to_tok = {i: t for t, i in vocab.items()}
        self.all_special_ids = [0]

    def encode(self, word: str, add_special_tokens: bool = False):
        if word in self._vocab:
            return [self._vocab[word]]
        # Simulate subword split for unknown multi-piece words.
        if " " in word or len(word) > 12:
            return [1, 2]
        # Invent an id for unseen short words.
        tid = max(self._vocab.values()) + 1 + abs(hash(word)) % 1000
        self._vocab[word] = tid
        self._id_to_tok[tid] = word
        return [tid]

    def convert_ids_to_tokens(self, tid: int):
        return self._id_to_tok.get(tid)

    def __len__(self):
        return len(self._id_to_tok)


def test_confidence_top_two_margin_binary():
    probs = np.array([[0.9, 0.1], [0.55, 0.45]])
    conf = confidence_from_class_probs(probs)
    np.testing.assert_allclose(conf, [0.8, 0.1])


def test_confidence_multiclass_and_difficulty():
    probs = np.array([[0.5, 0.3, 0.2], [0.34, 0.33, 0.33]])
    conf = confidence_from_class_probs(probs)
    np.testing.assert_allclose(conf, [0.2, 0.01], atol=1e-8)
    d = difficulty_from_confidence(conf)
    np.testing.assert_allclose(d, 1.0 - conf)


def test_pmd_rank_weights_sum_to_one():
    w = pmd_rank_weights(4, exponent=2.0)
    assert w.shape == (4,)
    np.testing.assert_allclose(w.sum(), 1.0)
    assert w[-1] > w[0]


def test_pmd_partition_counts_60_40():
    assert pmd_partition_counts(10, 0.6) == (6, 4)
    n_hard, n_easy = pmd_partition_counts(32, 0.6)
    assert n_hard + n_easy == 32
    assert n_hard > n_easy
    assert abs(n_hard / 32 - 0.6) <= 0.05


def test_pmd_sampling_probs_hard_prefers_low_confidence():
    conf = np.array([0.9, 0.1, 0.5, 0.2])  # idx1 hardest
    hard, easy, order = pmd_sampling_probs(conf, rank_exponent=2.0)
    np.testing.assert_allclose(hard.sum(), 1.0)
    np.testing.assert_allclose(easy.sum(), 1.0)
    # Descending confidence order: 0 (0.9), 2 (0.5), 3 (0.2), 1 (0.1)
    np.testing.assert_array_equal(order, [0, 2, 3, 1])
    assert hard[1] == max(hard)  # hardest gets highest hard weight
    assert easy[0] == max(easy)  # easiest gets highest easy weight


def test_pmd_batch_sampler_sizes():
    conf = np.linspace(0.1, 0.9, 20)
    gen = torch.Generator().manual_seed(0)
    sampler = PMDBatchSampler(
        conf, batch_size=10, hard_fraction=0.6, rank_exponent=2.0, generator=gen
    )
    batches = list(sampler)
    assert len(batches) == 2
    for b in batches:
        assert len(b) == 10
        assert all(0 <= i < 20 for i in b)


def test_select_verbalizers_unique_tokens():
    texts = [
        "sports ball game football soccer athlete",
        "sports stadium football match soccer",
        "politics election vote senate congress government",
        "politics vote parliament election democracy",
        "science physics atom molecule laboratory research",
        "science chemistry molecule laboratory experiment",
    ]
    y = np.array([0, 0, 1, 1, 2, 2])
    tok = _FakeTok(
        {
            "[PAD]": 0,
            "sports": 10,
            "football": 11,
            "soccer": 12,
            "politics": 13,
            "election": 14,
            "vote": 15,
            "science": 16,
            "molecule": 17,
            "laboratory": 18,
            "game": 19,
            "senate": 20,
            "physics": 21,
        }
    )
    sel = select_verbalizers(texts, y, tok)
    assert len(sel.tokens) == 3
    assert len(set(sel.tokens)) == 3
    assert len(set(sel.token_ids)) == 3
    assert sel.class_ids == [0, 1, 2]


def test_registry_self_adaptive_pmd():
    assert resolve_method_id("self_adaptive_pmd") == "self_adaptive_pmd"
    assert resolve_method_id("pmd") == "self_adaptive_pmd"
    assert REGISTRY["self_adaptive_pmd"] is SelfAdaptivePMDCurriculum
    kwargs = build_curriculum_kwargs(
        "self_adaptive_pmd",
        SimpleNamespace(
            curriculum_beta=0.5,
            hard_slice_quantile=0.8,
            random_state=42,
            sa_prompt_suffix=" This text is [MASK].",
            sa_hard_fraction=0.6,
            sa_rank_exponent=2.0,
            sa_score_batch_size=32,
            epochs=6,
        ),
    )
    assert kwargs["hard_fraction"] == 0.6
    assert kwargs["epochs"] == 6
    assert "prompt_suffix" in kwargs


def test_config_loader_sa_fields():
    from bio_is_curriculum.config.loader import merge_yaml_to_experiment_config

    cfg = merge_yaml_to_experiment_config(
        {
            "dataset": "webkb",
            "curriculum": {
                "method": "self_adaptive_pmd",
                "hard_fraction": 0.6,
                "rank_exponent": 2.0,
                "score_batch_size": 48,
                "prompt_suffix": " Label: [MASK].",
            },
            "training": {"epochs": 6},
        }
    )
    assert cfg.curriculum_method == "self_adaptive_pmd"
    assert cfg.sa_hard_fraction == 0.6
    assert cfg.sa_score_batch_size == 48
    assert cfg.sa_prompt_suffix == " Label: [MASK]."
