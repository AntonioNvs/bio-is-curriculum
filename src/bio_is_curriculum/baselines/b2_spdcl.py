"""Baseline 2 — Self-Paced Dynamic Curriculum Learning (Zhang et al. 2022).

SPDCL evaluates sample difficulty using linguistic features (nuclear norm of
token hidden states from the pretrained model) and model-capacity features
(delta nuclear norm between epochs). Each epoch re-sorts and re-bins the
training set with a progressive easy-to-hard pace.

Reference
---------
Zhang, X., Wang, J., Cheng, N., & Xiao, J. (2022).
Improving Imbalanced Text Classification with Dynamic Curriculum Learning.
arXiv:2210.14724. https://arxiv.org/abs/2210.14724
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from bio_is_curriculum.baselines.base import DynamicBaselineBase
from bio_is_curriculum.training.dynamic import run_dynamic_curriculum

if TYPE_CHECKING:
    from bio_is_curriculum.models.base import CurriculumModel
    from bio_is_curriculum.results.recorder import RunRecorder


class Baseline2SPDCL(DynamicBaselineBase):
    INDEX = 2
    NAME = "SPDCL (Zhang et al. 2022)"
    REFERENCE = (
        "Zhang, X., Wang, J., Cheng, N., & Xiao, J. (2022). "
        "Improving Imbalanced Text Classification with Dynamic Curriculum Learning. "
        "arXiv:2210.14724. https://arxiv.org/abs/2210.14724"
    )
    TRAINER_KIND = "dynamic"

    def __init__(
        self,
        model: Optional[CurriculumModel] = None,
        n_bins: int = 5,
        curriculum_epochs: int | None = None,
        anneal_epochs: int = 1,
        norm_subsample: int | None = None,
        hard_slice_quantile: float = 0.8,
        random_state: int = 42,
    ):
        self.model = model
        self.n_bins = n_bins
        self.curriculum_epochs = curriculum_epochs
        self.anneal_epochs = anneal_epochs
        self.norm_subsample = norm_subsample
        self.hard_slice_quantile = hard_slice_quantile
        self.random_state = random_state
        self.model_: CurriculumModel | None = None
        self.history_: list[dict] = []

    def fit(
        self,
        selector,
        X,
        y,
        X_test=None,
        y_test=None,
        X_val=None,
        y_val=None,
        X_text=None,
        X_val_text=None,
        X_test_text=None,
        recorder: Optional["RunRecorder"] = None,
    ):
        del selector, X  # SPDCL does not use BIOIS signals or TF-IDF features.
        if X_text is None:
            raise ValueError("SPDCL requires raw texts (ModernBERT backend).")
        if self.model is None:
            raise ValueError("SPDCL requires a CurriculumModel instance.")

        self.model_ = self.model
        X_eval = X_test_text if X_test_text is not None else X_test
        X_phase_val = X_val_text if X_val_text is not None else X_val

        self.history_ = run_dynamic_curriculum(
            self.model_,
            list(X_text),
            np.asarray(y),
            n_bins=self.n_bins,
            curriculum_epochs=self.curriculum_epochs,
            anneal_epochs=self.anneal_epochs,
            norm_subsample=self.norm_subsample,
            X_test=X_eval,
            y_test=y_test,
            X_val=X_phase_val,
            y_val=y_val,
            recorder=recorder,
            hard_slice_quantile=self.hard_slice_quantile,
            random_state=self.random_state,
        )
        return self
