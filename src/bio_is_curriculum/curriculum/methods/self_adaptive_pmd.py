"""Self-adaptive PMD curriculum (Feng et al., ACL SRW 2025 adaptation).

1. Score every train example with a frozen ModernBERT MLM + automatic verbalizers.
2. Train the classification head for a single full-data stage using PMD batches
   (60% hard-prioritized / 40% easy-prioritized squared-rank sampling).
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from bio_is_curriculum.curriculum.base import CurriculumBase
from bio_is_curriculum.models.logistic_regression import LogisticRegressionModel
from bio_is_curriculum.results.metrics import build_phase_metrics_row
from bio_is_curriculum.signals.self_adaptive import (
    DEFAULT_PROMPT_SUFFIX,
    VerbalizerSelection,
    difficulty_from_confidence,
    select_verbalizers,
    score_mlm_confidence,
)

if TYPE_CHECKING:
    from bio_is_curriculum.results.recorder import RunRecorder


class SelfAdaptivePMDCurriculum(CurriculumBase):
    """ACL SRW 2025 self-adaptive CL with PMD batch sampling."""

    METHOD_ID = "self_adaptive_pmd"
    REQUIRES_BIOIS = False

    def __init__(
        self,
        model=None,
        hard_slice_quantile: float = 0.8,
        random_state: int = 42,
        prompt_suffix: str = DEFAULT_PROMPT_SUFFIX,
        hard_fraction: float = 0.6,
        rank_exponent: float = 2.0,
        score_batch_size: int = 64,
        epochs: int | None = None,
        beta: float = 0.5,  # unused; accepted for registry common kwargs
        **_ignored,
    ):
        self.model = model
        self.hard_slice_quantile = hard_slice_quantile
        self.random_state = random_state
        self.prompt_suffix = prompt_suffix
        self.hard_fraction = hard_fraction
        self.rank_exponent = rank_exponent
        self.score_batch_size = score_batch_size
        self.epochs = epochs
        self.beta = beta
        self.confidence_: np.ndarray | None = None
        self.difficulty_: np.ndarray | None = None
        self.verbalizer_: VerbalizerSelection | None = None
        self.score_meta_: dict | None = None
        self.history_: list[dict] = []

    def _init_model(self):
        return (
            self.model
            if self.model is not None
            else LogisticRegressionModel(random_state=self.random_state)
        )

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
        if X_text is None:
            raise ValueError(
                "self_adaptive_pmd requires raw texts (use model: modernbert)."
            )
        if self.model is None:
            raise ValueError("self_adaptive_pmd requires a CurriculumModel instance.")

        y_arr = np.asarray(y)
        texts = list(X_text)
        self.model_ = self._init_model()

        if not hasattr(self.model_, "model_name"):
            raise ValueError(
                "self_adaptive_pmd requires a HF backbone with model_name "
                "(ModernBertModel)."
            )
        model_name = self.model_.model_name
        device = getattr(self.model_, "device", None)
        max_length = int(getattr(self.model_, "max_length", 256))

        # --- 1) Automatic verbalizers + frozen MLM confidence scoring ---
        t0_signals = time.perf_counter()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.verbalizer_ = select_verbalizers(texts, y_arr, tokenizer)
        confidence, meta = score_mlm_confidence(
            texts,
            self.verbalizer_,
            model_name=model_name,
            prompt_suffix=self.prompt_suffix,
            max_length=max_length,
            batch_size=self.score_batch_size,
            device=str(device) if device is not None else None,
        )
        self.confidence_ = confidence
        self.difficulty_ = difficulty_from_confidence(confidence)
        self.score_meta_ = meta
        signal_time = time.perf_counter() - t0_signals

        if recorder is not None:
            recorder.log_timing("sa_score_time_s", signal_time)
            recorder.log_timing("cl_signal_extract", signal_time)
            payload = {
                **meta,
                "hard_fraction": self.hard_fraction,
                "rank_exponent": self.rank_exponent,
                "score_batch_size": self.score_batch_size,
                "prompt_suffix": self.prompt_suffix,
            }
            with open(recorder.path("self_adaptive_verbalizers.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            # Persist per-example confidence for auditability.
            import csv

            with open(
                recorder.path("self_adaptive_scores.csv"),
                "w",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.DictWriter(
                    f, fieldnames=["idx", "y", "confidence", "difficulty"]
                )
                writer.writeheader()
                for i, (yi, c, d) in enumerate(
                    zip(y_arr, confidence, self.difficulty_)
                ):
                    writer.writerow(
                        {
                            "idx": i,
                            "y": int(yi),
                            "confidence": float(c),
                            "difficulty": float(d),
                        }
                    )
            recorder.update_config({"self_adaptive": payload})

        # --- 2) Single full-data PMD stage ---
        if self.epochs is not None and hasattr(self.model_, "epochs_per_stage"):
            self.model_.epochs_per_stage = int(self.epochs)
        if hasattr(self.model_, "set_phase"):
            self.model_.set_phase("pmd")

        sampling = {
            "strategy": "pmd",
            "confidence": confidence,
            "hard_fraction": float(self.hard_fraction),
            "rank_exponent": float(self.rank_exponent),
        }

        t0_train = time.perf_counter()
        X_val_input = X_val_text if X_val_text is not None else X_val
        fit_kwargs = dict(
            sample_weight=None,
            X_val=X_val_input,
            y_val=y_val,
            sampling=sampling,
        )
        # Older models may not accept sampling=; call signature checked below.
        try:
            self.model_.fit_stage(texts, y_arr, **fit_kwargs)
        except TypeError:
            fit_kwargs.pop("sampling", None)
            if hasattr(self.model_, "set_sampling"):
                self.model_.set_sampling(sampling)
            self.model_.fit_stage(
                texts,
                y_arr,
                sample_weight=None,
                X_val=X_val_input,
                y_val=y_val,
            )
        train_time = time.perf_counter() - t0_train

        history: list[dict] = []
        row = {
            "phase": "pmd",
            "n_samples": int(len(y_arr)),
            "n_train_samples": int(len(y_arr)),
            "n_classes_present": int(len(np.unique(y_arr))),
            "n_classes_total": int(len(np.unique(y_arr))),
            "n_classes_missing": 0,
            "n_rare_classes_pinned": 0,
            "n_iter": self.model_.n_iter,
            "train_time_s": float(train_time),
            "pred_time_s": float("nan"),
            "micro_f1": float("nan"),
            "macro_f1": float("nan"),
            "f1_weighted": float("nan"),
            "accuracy": float("nan"),
            "hard_slice_quantile": float(self.hard_slice_quantile),
            "hard_slice_macro_f1": float("nan"),
            "avg_seq_len": float("nan"),
            "compute_proxy": float("nan"),
            "best_val_macro_f1": float("nan"),
            "best_val_epoch": float("nan"),
            "steps_to_best_val": float("nan"),
            "phase_max_length": float("nan"),
        }

        if X_test is not None and y_test is not None:
            X_eval = X_test_text if X_test_text is not None else X_test
            t0_pred = time.perf_counter()
            proba = self.model_.predict_proba(X_eval)
            preds = np.argmax(proba, axis=1)
            pred_time = time.perf_counter() - t0_pred
            training_stats = {}
            if hasattr(self.model_, "get_training_stats"):
                training_stats = self.model_.get_training_stats()
            row = build_phase_metrics_row(
                phase="pmd",
                y_true=y_test,
                y_pred=preds,
                proba=proba,
                n_iter=self.model_.n_iter,
                train_time_s=train_time,
                pred_time_s=pred_time,
                hard_slice_quantile=self.hard_slice_quantile,
                training_stats=training_stats,
                balance_stats={
                    "n_train_samples": int(len(y_arr)),
                    "n_classes_present": int(len(np.unique(y_arr))),
                    "n_classes_total": int(len(np.unique(y_arr))),
                    "n_classes_missing": 0,
                    "n_rare_classes_pinned": 0,
                },
                n_train_instances=len(y_arr),
            )
            if recorder is not None:
                recorder.log_timing("metric_eval_time_s", pred_time)

        history.append(row)
        if recorder is not None:
            recorder.log_phase(row)
            recorder.log_timing("model_train_time_s", train_time)

        self.history_ = history
        return self
