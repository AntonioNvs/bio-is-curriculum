"""Tests for per-phase max_length in ModernBERT."""

import numpy as np

from bio_is_curriculum.models.modernbert import ModernBertModel, _DynamicPadCollator


def test_set_phase_updates_stage_max_length():
    model = ModernBertModel(max_length=256, random_state=0)
    model.set_phase("clean", max_length=96)
    assert model._phase_max_length == 96
    model.set_phase("hard", max_length=256)
    assert model._phase_max_length == 256


def test_dynamic_pad_collator_respects_max_length():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    collator = _DynamicPadCollator(tokenizer, max_length=32)
    batch = collator(
        [
            {"text": " ".join(["word"] * 100), "label": 0, "weight": 1.0},
            {"text": "short", "label": 1, "weight": 1.0},
        ]
    )
    assert batch["input_ids"].shape[1] <= 32
