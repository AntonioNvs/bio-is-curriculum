"""Single-fold experiment orchestrator."""

from __future__ import annotations

import os
import random
import time
import warnings
from collections import Counter
from types import SimpleNamespace

from bio_is_curriculum.config.cuda import configure_cuda_device

configure_cuda_device()

import numpy as np

from bio_is_curriculum.baselines import baseline_run_id, get_baseline
from bio_is_curriculum.baselines.b1_bengio2009 import Baseline1
from bio_is_curriculum.baselines.base import DynamicBaselineBase
from bio_is_curriculum.config.schema import ExperimentConfig
from bio_is_curriculum.curriculum.imbalance_losses import validate_imbalance_method
from bio_is_curriculum.curriculum.methods.registry import (
    build_curriculum_kwargs,
    get_curriculum_method,
    resolve_method_id,
)
from bio_is_curriculum.data.loader import DatasetLoader
from bio_is_curriculum.data.preprocessing import (
    maybe_upsample_for_biois,
    print_class_distribution,
    split_train_val,
    subsample_train_fraction,
)
from bio_is_curriculum.data.rare_class_upsampling import upsample_min_per_class
from bio_is_curriculum.models.logistic_regression import LogisticRegressionModel
from bio_is_curriculum.pipeline.modes import (
    CL_MODES,
    IS_MODES,
    normalize_mode,
    parse_is_baseline_index,
    uses_is_subset,
)
from bio_is_curriculum.results.recorder import RunRecorder
from bio_is_curriculum.selection.biois import BIOIS
from bio_is_curriculum.training.phased import eval_single_stage

BIOIS_STRATKFOLD_SPLITS = 5


def _is_transformer_backend(model: str) -> bool:
    return model in ("modernbert", "roberta")


def _normalize_model_backend(model: str) -> str:
    if model == "roberta":
        warnings.warn(
            "model='roberta' is deprecated; use model='modernbert'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return "modernbert"
    return model


def _build_model(cfg: ExperimentConfig, recorder: RunRecorder):
    if _is_transformer_backend(cfg.model):
        from bio_is_curriculum.models.modernbert import ModernBertModel

        return ModernBertModel(
            model_name=cfg.hf_model,
            epochs_per_stage=cfg.epochs_per_phase,
            batch_size=cfg.batch_size,
            eval_batch_size=cfg.eval_batch_size,
            max_length=cfg.max_length,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            imbalance_method=cfg.imbalance_method,
            class_balanced_loss=cfg.class_balanced_loss,
            effective_num_beta=cfg.effective_num_beta,
            dist_bal_tau=cfg.dist_bal_tau,
            dist_bal_logit_bias=cfg.dist_bal_logit_bias,
            aug_target_min_count=cfg.aug_target_min_count,
            aug_ratio=cfg.aug_ratio,
            aug_random_swap=cfg.aug_random_swap,
            aug_random_delete=cfg.aug_random_delete,
            random_state=cfg.random_state,
            history_callback=recorder.log_train_step,
        )
    if cfg.imbalance_method != "none":
        print("Warning: imbalance_method applies only to ModernBERT; ignored for LR.")
    return LogisticRegressionModel(random_state=cfg.random_state)


def _slice_selector_signals(selector, idx):
    return SimpleNamespace(
        _probaEveryone=selector._probaEveryone[idx],
        _y_proba_of_pred=selector._y_proba_of_pred[idx],
        _pred=selector._pred[idx],
    )


def _setup_recorder(cfg: ExperimentConfig) -> RunRecorder:
    mode = cfg.resolve_mode()
    if cfg.experiment_id is not None:
        base_dir = os.path.join(cfg.results_dir, cfg.experiment_id)
        run_id = f"{mode}_fold{cfg.fold}"
        return RunRecorder(base_dir=base_dir, run_id=run_id)

    recorder = RunRecorder(base_dir=cfg.results_dir, run_id=cfg.run_id)
    if cfg.run_id is None:
        old_dir = recorder.run_dir
        new_id = f"{mode}-{recorder.run_id}"
        new_dir = os.path.join(cfg.results_dir, new_id)
        os.rename(old_dir, new_dir)
        recorder.run_dir = new_dir
        recorder.run_id = new_id
    return recorder


def _build_curriculum_kwargs(cfg: ExperimentConfig) -> dict:
    ns = SimpleNamespace(
        curriculum_beta=cfg.curriculum_beta,
        hard_slice_quantile=cfg.hard_slice_quantile,
        random_state=cfg.random_state,
        curriculum_q=cfg.curriculum_q,
        curriculum_n_steps=cfg.curriculum_n_steps,
        curriculum_alpha_decay=cfg.curriculum_alpha_decay,
        curriculum_soft_lambda_init=cfg.curriculum_soft_lambda_init,
        curriculum_soft_lambda_growth=cfg.curriculum_soft_lambda_growth,
        curriculum_soft_lambda_max=cfg.curriculum_soft_lambda_max,
        curriculum_soft_min_weight=cfg.curriculum_soft_min_weight,
        curriculum_soft_stability_tol=cfg.curriculum_soft_stability_tol,
        curriculum_soft_saturation_patience=cfg.curriculum_soft_saturation_patience,
        curriculum_soft_max_effective_steps=cfg.curriculum_soft_max_effective_steps,
        curriculum_loss_scheme=cfg.curriculum_loss_scheme,
        curriculum_lambda_init=cfg.curriculum_lambda_init,
        curriculum_lambda_mult=cfg.curriculum_lambda_mult,
        curriculum_lambda_step=cfg.curriculum_lambda_step,
        curriculum_lambda_max=cfg.curriculum_lambda_max,
        curriculum_lambda2=cfg.curriculum_lambda2,
        curriculum_loss_prior_reliability=cfg.curriculum_loss_prior_reliability,
        curriculum_min_weight=cfg.curriculum_min_weight,
        curriculum_loss_recompute_every=cfg.curriculum_loss_recompute_every,
        td_probe_epochs=cfg.td_probe_epochs,
        td_metric=cfg.td_metric,
    )
    return build_curriculum_kwargs(cfg.resolve_curriculum_method(), ns)


def run_experiment(cfg: ExperimentConfig) -> dict:
    """Execute one fold and return final metrics."""
    cfg.mode = normalize_mode(cfg.mode)
    cfg.model = _normalize_model_backend(cfg.model)
    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    cfg.imbalance_method = validate_imbalance_method(cfg.imbalance_method)
    if cfg.class_balanced_loss is not None:
        cfg.imbalance_method = "inverse_freq_ce" if cfg.class_balanced_loss else "none"

    method = cfg.resolve_curriculum_method()
    cfg.curriculum_method = resolve_method_id(method)

    is_baseline_idx = parse_is_baseline_index(cfg.mode)
    baseline_cls = None
    if cfg.baseline is not None:
        baseline_cls = get_baseline(cfg.baseline)
        cfg.mode = baseline_run_id(cfg.baseline)
    elif is_baseline_idx is not None:
        baseline_cls = get_baseline(is_baseline_idx)

    curriculum_cls = (
        baseline_cls
        if baseline_cls is not None
        else get_curriculum_method(cfg.curriculum_method)
    )

    recorder = _setup_recorder(cfg)
    print("=" * 50)
    print(f"run_id : {recorder.run_id}")
    print(f"mode   : {cfg.mode}")
    print(f"model  : {cfg.model}")
    print(f"results: {recorder.run_dir}")
    print("=" * 50)

    recorder.save_config(cfg.to_dict())
    t0_total = time.perf_counter()

    loader = DatasetLoader(data_dir=cfg.data_dir, dataset_name=cfg.dataset)
    t0_load = time.perf_counter()
    texts_train_raw = texts_test = None
    y_texts_raw = y_test_texts = None

    if _is_transformer_backend(cfg.model):
        (
            X_train_raw, y_train_raw, X_test, y_test,
            texts_train_raw, texts_test,
        ) = loader.load_aligned_fold(cfg.fold, n_splits=cfg.n_splits)
        y_texts_raw = y_train_raw
        y_test_texts = y_test
    else:
        X_train_raw, y_train_raw, X_test, y_test = loader.load_tfidf_fold(cfg.fold)

    split = split_train_val(
        X_train_raw, y_train_raw, texts_train_raw, y_texts_raw,
        random_state=cfg.random_state,
    )
    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]
    texts_train, texts_val = split["texts_train"], split["texts_val"]
    y_texts_train, y_texts_val = split["y_texts_train"], split["y_texts_val"]

    if cfg.train_fraction < 1.0:
        t0_sub = time.perf_counter()
        sub = subsample_train_fraction(
            X_train, y_train, texts_train, y_texts_train,
            fraction=cfg.train_fraction,
            random_state=cfg.random_state,
        )
        X_train, y_train = sub["X_train"], sub["y_train"]
        texts_train, y_texts_train = sub["texts_train"], sub["y_texts_train"]
        recorder.save_train_subsample(
            train_fraction=cfg.train_fraction,
            n_train_before=sub["n_before"],
            n_train_after=sub["n_after"],
        )
        print(
            f"  Train subsample ({cfg.train_fraction:.0%}): "
            f"{sub['n_before']} -> {sub['n_after']}"
        )
        recorder.log_timing("train_subsample_time_s", time.perf_counter() - t0_sub)

    needs_is = cfg.mode in IS_MODES or is_baseline_idx is not None
    needs_biois_for_cl = (
        cfg.mode in CL_MODES
        and getattr(curriculum_cls, "REQUIRES_BIOIS", True)
    )
    needs_biois_for_baseline = (
        baseline_cls is not None
        and not issubclass(baseline_cls, DynamicBaselineBase)
        and getattr(baseline_cls, "REQUIRES_BIOIS", True)
    )
    needs_biois = needs_is or needs_biois_for_cl or needs_biois_for_baseline
    if needs_biois:
        X_train, y_train, texts_train, y_texts_train = maybe_upsample_for_biois(
            X_train, y_train, texts_train, cfg.random_state,
        )

    print_class_distribution(y_train)
    recorder.log_timing("data_load_time_s", time.perf_counter() - t0_load)

    selector = None
    preprocess_time = 0.0
    if needs_biois:
        biois_beta = cfg.beta if needs_is else 0.0
        biois_theta = cfg.theta if needs_is else 0.0
        selector = BIOIS(beta=biois_beta, theta=biois_theta, random_state=cfg.random_state)
        t0_is = time.perf_counter()
        selector.fit(X_train, y_train)
        preprocess_time = time.perf_counter() - t0_is
        recorder.log_timing("is_fit_time_s", preprocess_time)

        y_train_arr = np.asarray(y_train)
        selected_mask = np.zeros(len(y_train_arr), dtype=bool)
        selected_mask[selector.sample_indices_] = True
        removed_y = y_train_arr[~selected_mask]
        total_by_class = Counter(y_train_arr.tolist())
        removed_by_class = Counter(removed_y.tolist())

        recorder.save_instance_selection(
            n_train_before=len(y_train_arr),
            n_train_after=len(selector.sample_indices_) if needs_is else len(y_train_arr),
            reduction=selector.reduction_ if needs_is else 0.0,
            beta=biois_beta,
            theta=biois_theta,
            removed_by_class=dict(removed_by_class),
            total_by_class=dict(total_by_class),
        )
        n_after_is = len(selector.sample_indices_) if needs_is else len(y_train_arr)
        recorder.update_config({
            "is_stats": {
                "n_train_before": len(y_train_arr),
                "n_train_after": n_after_is,
                "n_removed": len(y_train_arr) - n_after_is,
                "reduction": selector.reduction_ if needs_is else 0.0,
            },
        })

    recorder.log_timing("preprocess_time_s", preprocess_time)
    model = _build_model(cfg, recorder)

    y_all = np.concatenate([
        np.asarray(y_texts_train if y_texts_train is not None else y_train),
        np.asarray(y_test_texts if y_test_texts is not None else y_test),
    ])
    if hasattr(model, "num_labels"):
        model.num_labels = int(np.max(y_all)) + 1

    metrics: dict = {}

    if cfg.mode == "raw":
        y_tr = y_texts_train if y_texts_train is not None else y_train
        y_te = y_test_texts if y_test_texts is not None else y_test
        X_train_input = texts_train if texts_train else X_train
        X_val_input = texts_val if texts_val else X_val
        X_test_input = texts_test if texts_test else X_test
        y_val_input = y_texts_val if y_texts_val is not None else y_val
        if hasattr(model, "set_phase"):
            model.set_phase("full")
        if hasattr(model, "epochs_per_stage"):
            model.epochs_per_stage = cfg.epochs
        t0_train = time.perf_counter()
        model.fit_stage(X_train_input, y_tr, X_val=X_val_input, y_val=y_val_input)
        train_time = time.perf_counter() - t0_train
        recorder.log_timing("model_train_time_s", train_time)
        _, _, metrics = eval_single_stage(
            model, X_test_input, y_te, recorder,
            train_time=train_time,
            hard_slice_quantile=cfg.hard_slice_quantile,
            n_train_instances=len(y_tr),
        )

    elif cfg.mode == "is":
        idx = selector.sample_indices_
        X_sub = texts_train if texts_train else X_train
        X_train_input = [X_sub[i] for i in idx] if isinstance(X_sub, list) else X_sub[idx]
        y_src = y_texts_train if y_texts_train is not None else y_train
        y_te = y_test_texts if y_test_texts is not None else y_test
        y_sub = np.asarray(y_src[idx])
        X_train_input, y_sub, _, _ = upsample_min_per_class(
            X_train_input, y_sub, min_count=BIOIS_STRATKFOLD_SPLITS,
            random_state=cfg.random_state,
        )
        if hasattr(model, "epochs_per_stage"):
            model.epochs_per_stage = cfg.epochs
        t0_train = time.perf_counter()
        model.fit_stage(
            X_train_input, y_sub,
            X_val=texts_val if texts_val else X_val,
            y_val=y_texts_val if y_texts_val is not None else y_val,
        )
        train_time = time.perf_counter() - t0_train
        recorder.log_timing("model_train_time_s", train_time)
        _, _, metrics = eval_single_stage(
            model, texts_test if texts_test else X_test, y_te, recorder,
            train_time=train_time,
            hard_slice_quantile=cfg.hard_slice_quantile,
            n_train_instances=len(y_sub),
        )

    else:
        y_src = y_texts_train if y_texts_train is not None else y_train
        y_te = y_test_texts if y_test_texts is not None else y_test

        if uses_is_subset(cfg.mode):
            idx = selector.sample_indices_
            cl_selector = _slice_selector_signals(selector, idx)
            X_cl = X_train[idx]
            y_cl = y_src[idx]
            texts_cl = [texts_train[i] for i in idx] if texts_train else None
        else:
            cl_selector = selector
            X_cl = X_train
            y_cl = y_src
            texts_cl = texts_train

        if uses_is_subset(cfg.mode):
            y_ic = np.asarray(y_cl)
            if texts_cl is not None:
                X_cl, y_ic, st_post, texts_cl = upsample_min_per_class(
                    X_cl, y_ic, min_count=BIOIS_STRATKFOLD_SPLITS,
                    random_state=cfg.random_state, texts=texts_cl,
                )
            else:
                X_cl, y_ic, st_post, _ = upsample_min_per_class(
                    X_cl, y_ic, min_count=BIOIS_STRATKFOLD_SPLITS,
                    random_state=cfg.random_state,
                )
            y_cl = y_ic
            if st_post.n_added > 0:
                dup = st_post.dup_row_idx
                cl_selector = SimpleNamespace(
                    _probaEveryone=np.concatenate(
                        [cl_selector._probaEveryone, cl_selector._probaEveryone[dup]]
                    ),
                    _y_proba_of_pred=np.concatenate(
                        [cl_selector._y_proba_of_pred, cl_selector._y_proba_of_pred[dup]]
                    ),
                    _pred=np.concatenate([cl_selector._pred, cl_selector._pred[dup]]),
                )

        if hasattr(model, "epochs_per_stage"):
            model.epochs_per_stage = cfg.epochs_per_phase

        if baseline_cls is not None:
            if issubclass(baseline_cls, DynamicBaselineBase):
                curriculum = baseline_cls(
                    model=model,
                    n_bins=cfg.spdcl_n_bins,
                    curriculum_epochs=cfg.spdcl_curriculum_epochs,
                    anneal_epochs=cfg.spdcl_anneal_epochs,
                    norm_subsample=cfg.spdcl_norm_subsample,
                    hard_slice_quantile=cfg.hard_slice_quantile,
                    random_state=cfg.random_state,
                )
            else:
                if baseline_cls is Baseline1:
                    curriculum = baseline_cls(
                        model=model,
                        easy_fraction=cfg.b1_easy_fraction,
                        use_global_quantile=cfg.b1_use_global_quantile,
                        hard_slice_quantile=cfg.hard_slice_quantile,
                        random_state=cfg.random_state,
                    )
                else:
                    q_low, q_mid, q_high = cfg.curriculum_q
                    curriculum = baseline_cls(
                        model=model,
                        beta=cfg.curriculum_beta,
                        q_low=q_low, q_mid=q_mid, q_high=q_high,
                        hard_slice_quantile=cfg.hard_slice_quantile,
                        random_state=cfg.random_state,
                    )
        else:
            CurriculumCls = get_curriculum_method(cfg.curriculum_method)
            curriculum_kwargs = _build_curriculum_kwargs(cfg)
            curriculum_kwargs["model"] = model
            curriculum = CurriculumCls(**curriculum_kwargs)

        curriculum.fit(
            cl_selector, X_cl, y_cl,
            X_test=X_test, y_test=y_te,
            X_val=X_val, y_val=y_texts_val if y_texts_val is not None else y_val,
            X_text=texts_cl, X_val_text=texts_val, X_test_text=texts_test,
            recorder=recorder,
        )
        model = curriculum.model_
        X_test_input = texts_test if texts_test else X_test
        proba = model.predict_proba(X_test_input)
        preds = np.argmax(proba, axis=1)
        metrics = curriculum.history_[-1] if curriculum.history_ else {}
        y_save = y_test_texts if y_test_texts is not None else y_test
        recorder.save_predictions(y_save, preds, proba)
        recorder.log_timing("total_run_time_s", time.perf_counter() - t0_total)
        print(f"Macro-F1: {metrics.get('macro_f1', float('nan')):.4f}")
        return metrics

    X_test_input = texts_test if texts_test else X_test
    y_save = y_test_texts if y_test_texts is not None else y_test
    proba = model.predict_proba(X_test_input)
    preds = np.argmax(proba, axis=1)
    recorder.save_predictions(y_save, preds, proba)
    recorder.log_timing("total_run_time_s", time.perf_counter() - t0_total)
    print(f"Macro-F1: {metrics.get('macro_f1', float('nan')):.4f}")
    return metrics
