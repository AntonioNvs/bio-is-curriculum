"""CLI argument parser for single-fold runs."""

from __future__ import annotations

import argparse

from bio_is_curriculum.config.defaults import DEFAULTS
from bio_is_curriculum.config.schema import ExperimentConfig
from bio_is_curriculum.curriculum.imbalance_losses import IMBALANCE_METHODS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="biO-IS-Curriculum: instance selection + curriculum learning for text classification."
    )
    p.add_argument("dataset", type=str)
    p.add_argument("--data_dir", type=str, default="datasets")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-splits", dest="n_splits", type=int, default=10)
    p.add_argument(
        "--mode",
        choices=[
            "raw", "is", "cl", "is_cl", "is_continuous_cl", "is_continuos_cl", "is_b2",
        ],
        default="is_cl",
    )
    p.add_argument("--baseline", type=int, default=None)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--theta", type=float, default=0.2)
    p.add_argument("--random-state", dest="random_state", type=int, default=42)
    p.add_argument("--curriculum-method", dest="curriculum_method", type=str, default=None)
    p.add_argument("--curriculum-beta", dest="curriculum_beta", type=float, default=0.5)
    p.add_argument(
        "--curriculum-q", dest="curriculum_q", type=float, nargs=3,
        default=(0.3, 0.6, 0.95),
    )
    p.add_argument("--curriculum-n-steps", dest="curriculum_n_steps", type=int, default=6)
    p.add_argument("--curriculum-alpha-decay", dest="curriculum_alpha_decay", type=float, default=10.0)
    p.add_argument("--curriculum-soft-lambda-init", dest="curriculum_soft_lambda_init", type=float, default=0.25)
    p.add_argument("--curriculum-soft-lambda-growth", dest="curriculum_soft_lambda_growth", type=float, default=1.4)
    p.add_argument("--curriculum-soft-lambda-max", dest="curriculum_soft_lambda_max", type=float, default=1.0)
    p.add_argument("--curriculum-soft-min-weight", dest="curriculum_soft_min_weight", type=float, default=1e-3)
    p.add_argument("--curriculum-soft-stability-tol", dest="curriculum_soft_stability_tol", type=float, default=5e-3)
    p.add_argument("--curriculum-soft-saturation-patience", dest="curriculum_soft_saturation_patience", type=int, default=2)
    p.add_argument("--curriculum-soft-max-effective-steps", dest="curriculum_soft_max_effective_steps", type=int, default=6)
    p.add_argument("--curriculum-loss-scheme", dest="curriculum_loss_scheme", choices=("binary", "linear", "log", "mixture"), default="linear")
    p.add_argument("--curriculum-lambda-init", dest="curriculum_lambda_init", type=float, default=0.5)
    p.add_argument("--curriculum-lambda-step", dest="curriculum_lambda_step", type=float, default=0.5)
    p.add_argument("--curriculum-lambda-mult", dest="curriculum_lambda_mult", type=float, default=1.0)
    p.add_argument("--curriculum-lambda-max", dest="curriculum_lambda_max", type=float, default=None)
    p.add_argument("--curriculum-lambda2", dest="curriculum_lambda2", type=float, default=None)
    p.add_argument("--curriculum-loss-prior-reliability", dest="curriculum_loss_prior_reliability", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--curriculum-min-weight", dest="curriculum_min_weight", type=float, default=1e-3)
    p.add_argument("--curriculum-loss-recompute-every", dest="curriculum_loss_recompute_every", type=int, default=2)
    p.add_argument("--td-probe-epochs", dest="td_probe_epochs", type=int, default=2)
    p.add_argument(
        "--td-metric",
        dest="td_metric",
        choices=("confidence", "variability"),
        default="confidence",
    )
    p.add_argument("--spdcl-n-bins", dest="spdcl_n_bins", type=int, default=5)
    p.add_argument("--spdcl-curriculum-epochs", dest="spdcl_curriculum_epochs", type=int, default=None)
    p.add_argument("--spdcl-anneal-epochs", dest="spdcl_anneal_epochs", type=int, default=1)
    p.add_argument("--spdcl-norm-subsample", dest="spdcl_norm_subsample", type=int, default=None)
    p.add_argument("--model", choices=["lr", "modernbert", "roberta"], default="modernbert")
    p.add_argument("--hf-model", dest="hf_model", type=str, default="answerdotai/ModernBERT-base")
    p.add_argument("--train-fraction", dest="train_fraction", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--epochs-per-phase", dest="epochs_per_phase", type=int, default=2)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, default=64)
    p.add_argument("--max-length", dest="max_length", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-3)
    p.add_argument("--warmup-ratio", dest="warmup_ratio", type=float, default=0.06)
    p.add_argument("--imbalance-method", dest="imbalance_method", choices=IMBALANCE_METHODS, default="inverse_freq_ce")
    p.add_argument("--effective-num-beta", dest="effective_num_beta", type=float, default=0.9999)
    p.add_argument("--dist-bal-tau", dest="dist_bal_tau", type=float, default=1.0)
    p.add_argument("--dist-bal-logit-bias", dest="dist_bal_logit_bias", type=float, default=0.1)
    p.add_argument("--aug-target-min-count", dest="aug_target_min_count", type=int, default=5)
    p.add_argument("--aug-ratio", dest="aug_ratio", type=float, default=0.25)
    p.add_argument("--aug-random-swap", dest="aug_random_swap", type=float, default=0.2)
    p.add_argument("--aug-random-delete", dest="aug_random_delete", type=float, default=0.1)
    p.add_argument("--class-balanced-loss", dest="class_balanced_loss", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--hard-slice-quantile", dest="hard_slice_quantile", type=float, default=0.8)
    p.add_argument(
        "--cuda-device-id",
        dest="cuda_device_id",
        type=int,
        default=DEFAULTS["cuda_device_id"],
        help="Physical GPU index (sets CUDA_VISIBLE_DEVICES when unset). Default: 7.",
    )
    p.add_argument("--results-dir", dest="results_dir", type=str, default="results")
    p.add_argument("--experiment-id", dest="experiment_id", type=str, default=None)
    p.add_argument("--run-id", dest="run_id", type=str, default=None)
    return p


def args_to_config(args: argparse.Namespace) -> ExperimentConfig:
    data = vars(args).copy()
    data["curriculum_q"] = tuple(args.curriculum_q)
    return ExperimentConfig.from_dict(data)
