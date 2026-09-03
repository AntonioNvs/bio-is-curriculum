"""Dynamic per-epoch curriculum training (SPDCL and similar)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from bio_is_curriculum.signals.nuclear_norm import NuclearNormScorer
from bio_is_curriculum.training.phased import eval_single_stage

if TYPE_CHECKING:
    from bio_is_curriculum.models.base import CurriculumModel
    from bio_is_curriculum.results.recorder import RunRecorder


def scatter_into_bins(sorted_indices: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Interleave sorted samples into n_bins shares (SPDCL Figure 5)."""
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for rank, idx in enumerate(sorted_indices):
        bins[rank % n_bins].append(int(idx))
    return [np.array(b, dtype=int) for b in bins if len(b) > 0]


def progressive_bin_indices(bins: list[np.ndarray], epoch: int) -> np.ndarray:
    """Union of bins 0..epoch (inclusive), capped at all bins."""
    visible = bins[: min(epoch + 1, len(bins))]
    if not visible:
        return np.array([], dtype=int)
    return np.concatenate(visible)


def _subsample_indices(n: int, max_samples: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_samples is None or max_samples >= n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_samples, replace=False))


def run_dynamic_curriculum(
    model: CurriculumModel,
    texts: list[str],
    y: np.ndarray,
    *,
    n_bins: int,
    curriculum_epochs: int | None = None,
    anneal_epochs: int = 1,
    norm_subsample: int | None = None,
    X_test=None,
    y_test=None,
    X_val=None,
    y_val=None,
    recorder: RunRecorder | None = None,
    hard_slice_quantile: float = 0.8,
    random_state: int = 42,
) -> list[dict]:
    """SPDCL-style dynamic reordering and resampling each epoch."""
    y = np.asarray(y)
    n = len(y)
    history: list[dict] = []
    rng = np.random.default_rng(random_state)

    cur_epochs = curriculum_epochs if curriculum_epochs is not None else n_bins
    total_epochs = cur_epochs + max(0, anneal_epochs)
    nuclear_norm_total = 0.0
    t0_total = time.perf_counter()

    scorer = NuclearNormScorer()
    # Linguistic difficulty from pretrained backbone BEFORE any gradient step.
    t0_norm = time.perf_counter()
    hidden = model.extract_hidden_states(texts)
    scorer.score_pretrain(hidden)
    nuclear_norm_total += time.perf_counter() - t0_norm

    for epoch in range(total_epochs):
        if epoch < cur_epochs:
            if epoch == 0:
                difficulty = scorer.difficulty_for_epoch(scorer.initial_norms, curriculum_epoch=0)
                order = np.argsort(difficulty)
            else:
                t0_norm = time.perf_counter()
                norm_idx = _subsample_indices(n, norm_subsample, rng)
                norm_texts = [texts[i] for i in norm_idx]
                hidden = model.extract_hidden_states(norm_texts)
                current = scorer.score_current(hidden)
                if norm_subsample is not None:
                    full_current = np.zeros(n, dtype=np.float64)
                    full_current[norm_idx] = current
                    current = full_current
                difficulty = scorer.score_delta(current)
                nuclear_norm_total += time.perf_counter() - t0_norm
                order = np.argsort(-difficulty)

            bins = scatter_into_bins(order, n_bins)
            active_idx = progressive_bin_indices(bins, epoch)
            phase_name = f"spdcl_bin_{epoch + 1}"
        else:
            active_idx = np.arange(n)
            phase_name = f"spdcl_anneal_{epoch - cur_epochs + 1}"

        X_phase = [texts[i] for i in active_idx]
        y_phase = y[active_idx]
        weights = np.ones(len(active_idx), dtype=np.float64)

        if hasattr(model, "set_phase"):
            model.set_phase(phase_name)

        t0_train = time.perf_counter()
        model.fit_epoch(
            X_phase,
            y_phase,
            sample_weight=weights,
            X_val=X_val,
            y_val=y_val,
            epoch_seed=random_state + epoch,
        )
        train_time = time.perf_counter() - t0_train

        _, _, row = eval_single_stage(
            model,
            X_test,
            y_test,
            recorder,
            phase=phase_name,
            train_time=train_time,
            hard_slice_quantile=hard_slice_quantile,
            n_train_instances=len(active_idx),
        )
        history.append(row)

    if recorder is not None:
        recorder.log_timing("nuclear_norm_time_s", nuclear_norm_total)
        recorder.log_timing("model_train_time_s", time.perf_counter() - t0_total)

    return history
