"""Multi-fold YAML experiment runner."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from bio_is_curriculum.cli.docker_launcher import docker_run
from bio_is_curriculum.config.campaign import expand_campaign, resolve_campaign_timestamp
from bio_is_curriculum.config.cuda import running_in_docker
from bio_is_curriculum.config.defaults import DEFAULTS
from bio_is_curriculum.config.loader import load_experiment_spec
from bio_is_curriculum.config.schema import BatchExperimentConfig, DockerConfig
from bio_is_curriculum.data.loader import DatasetLoader
from bio_is_curriculum.results.aggregator import aggregate, print_summary
from bio_is_curriculum.results.manifest import (
    ExperimentManifest,
    build_run_entry,
    docker_to_dict,
    resolve_event_description,
    resolve_summary_config,
    utc_now_iso,
    write_manifest,
)

_DISCRETE_METHODS = frozenset({
    "biois_discrete",
    "length_discrete",
    "loss_discrete",
    "tfidf_discrete",
})


@dataclass
class BatchRunResult:
    experiment_id: str
    batch: BatchExperimentConfig
    failed: list[tuple[str, int, int]]

    @property
    def status(self) -> str:
        return "failed" if self.failed else "ok"


def _discover_folds(dataset: str, data_dir: str, n_splits: int) -> list[int]:
    df = DatasetLoader(data_dir, dataset).load_splits(n_splits=n_splits)
    return sorted(df["fold_id"].tolist())


def _build_cli_args(cfg, mode: str, fold: int, experiment_id: str) -> list[str]:
    args = [
        sys.executable, "-m", "bio_is_curriculum.cli.main",
        cfg.dataset,
        "--fold", str(fold),
        "--experiment-id", experiment_id,
        "--results-dir", cfg.results_dir,
        "--n-splits", str(cfg.n_splits),
        "--data_dir", cfg.data_dir,
        "--model", cfg.model,
        "--random-state", str(cfg.random_state),
        "--hard-slice-quantile", str(cfg.hard_slice_quantile),
    ]
    baseline_match = re.match(r"^b([0-9]+)$", mode)
    if baseline_match:
        args += ["--baseline", baseline_match.group(1)]
    else:
        args += ["--mode", mode]

    if mode not in ("raw",):
        args += ["--beta", str(cfg.beta), "--theta", str(cfg.theta)]

    args += [
        "--epochs", str(cfg.epochs),
        "--epochs-per-phase", str(cfg.epochs_per_phase),
        "--batch-size", str(cfg.batch_size),
        "--eval-batch-size", str(cfg.eval_batch_size),
        "--max-length", str(cfg.max_length),
        "--lr", str(cfg.lr),
        "--weight-decay", str(cfg.weight_decay),
        "--warmup-ratio", str(cfg.warmup_ratio),
        "--imbalance-method", str(cfg.imbalance_method),
        "--spdcl-n-bins", str(cfg.spdcl_n_bins),
        "--spdcl-anneal-epochs", str(cfg.spdcl_anneal_epochs),
        "--cuda-device-id", str(cfg.cuda_device_id),
    ]
    if cfg.spdcl_curriculum_epochs is not None:
        args += ["--spdcl-curriculum-epochs", str(cfg.spdcl_curriculum_epochs)]
    if cfg.spdcl_norm_subsample is not None:
        args += ["--spdcl-norm-subsample", str(cfg.spdcl_norm_subsample)]
    if cfg.model == "roberta":
        args += ["--hf-model", cfg.hf_model]

    if mode not in ("raw", "is"):
        args += ["--curriculum-method", cfg.curriculum_method]
        args += ["--curriculum-beta", str(cfg.curriculum_beta)]
        q_low, q_mid, q_high = cfg.curriculum_q
        if cfg.curriculum_method in _DISCRETE_METHODS:
            args += ["--curriculum-q", str(q_low), str(q_mid), str(q_high)]
        if cfg.curriculum_method == "spcl_soft":
            args += [
                "--curriculum-n-steps", str(cfg.curriculum_n_steps),
                "--curriculum-alpha-decay", str(cfg.curriculum_alpha_decay),
                "--curriculum-soft-lambda-init", str(cfg.curriculum_soft_lambda_init),
                "--curriculum-soft-lambda-growth", str(cfg.curriculum_soft_lambda_growth),
                "--curriculum-soft-lambda-max", str(cfg.curriculum_soft_lambda_max),
                "--curriculum-soft-min-weight", str(cfg.curriculum_soft_min_weight),
                "--curriculum-soft-stability-tol", str(cfg.curriculum_soft_stability_tol),
                "--curriculum-soft-saturation-patience",
                str(cfg.curriculum_soft_saturation_patience),
                "--curriculum-soft-max-effective-steps",
                str(cfg.curriculum_soft_max_effective_steps),
            ]
        if cfg.curriculum_method == "spcl_loss":
            args += [
                "--curriculum-n-steps", str(cfg.curriculum_n_steps),
                "--curriculum-loss-scheme", str(cfg.curriculum_loss_scheme),
                "--curriculum-lambda-init", str(cfg.curriculum_lambda_init),
                "--curriculum-lambda-step", str(cfg.curriculum_lambda_step),
                "--curriculum-lambda-mult", str(cfg.curriculum_lambda_mult),
                "--curriculum-min-weight", str(cfg.curriculum_min_weight),
                "--curriculum-loss-recompute-every",
                str(cfg.curriculum_loss_recompute_every),
            ]
            if cfg.curriculum_lambda_max is not None:
                args += ["--curriculum-lambda-max", str(cfg.curriculum_lambda_max)]
            if cfg.curriculum_lambda2 is not None:
                args += ["--curriculum-lambda2", str(cfg.curriculum_lambda2)]
            args += [
                "--curriculum-loss-prior-reliability",
                str(cfg.curriculum_loss_prior_reliability).lower(),
            ]

    return args


def _resolve_batches(
    spec,
    *,
    dataset_override: str | None,
) -> tuple[list[BatchExperimentConfig], str]:
    if spec.campaign is not None:
        timestamp = resolve_campaign_timestamp(spec.campaign)
        return expand_campaign(spec.campaign, timestamp=timestamp), timestamp
    assert spec.batch is not None
    return [spec.batch], datetime.now().strftime("%Y%m%d-%H%M%S")


def _apply_cli_overrides(
    batches: list[BatchExperimentConfig],
    *,
    dataset: str | None,
    folds: list[int] | None,
) -> list[BatchExperimentConfig]:
    updated: list[BatchExperimentConfig] = []
    for batch in batches:
        if dataset and batch.dataset != dataset:
            continue
        if folds is not None:
            batch.folds = folds
        updated.append(batch)
    return updated


def _run_batch(
    batch: BatchExperimentConfig,
    *,
    verbose: bool,
    fail_fast: bool,
    dry_run: bool,
) -> BatchRunResult:
    cfg = batch.run
    cfg.dataset = batch.dataset or cfg.dataset
    cfg.n_splits = batch.n_splits

    folds = sorted(batch.folds) if batch.folds else _discover_folds(
        cfg.dataset, cfg.data_dir, cfg.n_splits
    )
    modes = batch.modes
    experiment_id = batch.experiment_id or (
        f"{cfg.dataset}-{cfg.n_splits}cv-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    )
    experiment_dir = os.path.join(cfg.results_dir, experiment_id)
    if not dry_run:
        os.makedirs(experiment_dir, exist_ok=True)

    print("=" * 65)
    print(f"  Dataset : {cfg.dataset}  ({cfg.n_splits}-fold CV)")
    print(f"  Modes   : {', '.join(modes)}")
    print(f"  Method  : {cfg.curriculum_method}")
    print(f"  Folds   : {folds}")
    print(f"  Exp ID  : {experiment_id}")
    print("=" * 65)

    failed: list[tuple[str, int, int]] = []
    run_num = 0
    total = len(modes) * len(folds)
    for mode in modes:
        for fold in folds:
            run_num += 1
            label = f"[{run_num}/{total}] {mode} fold={fold}"
            cli_args = _build_cli_args(cfg, mode, fold, experiment_id)
            if dry_run:
                print(f"{label}: {' '.join(cli_args)}")
                continue
            print(f"\n{label} ", end="", flush=True)
            result = subprocess.run(
                cli_args, check=False, capture_output=not verbose, text=True
            )
            if result.returncode != 0:
                print("FAIL")
                failed.append((mode, fold, result.returncode))
                if fail_fast:
                    return BatchRunResult(experiment_id, batch, failed)
            else:
                print("OK")

    if not dry_run:
        summary_df = aggregate(experiment_dir, modes, folds)
        summary_path = os.path.join(experiment_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False, float_format="%.6f")
        print_summary(summary_df)
        print(f"\nSummary: {summary_path}")

    return BatchRunResult(experiment_id, batch, failed)


def main():
    parser = argparse.ArgumentParser(description="biO-IS-Curriculum YAML experiment runner.")
    parser.add_argument("config", type=str)
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--docker", action="store_true", help="Run inside Docker on the host.")
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Run locally (used inside containers).",
    )
    parser.add_argument("--docker-gpu", type=int, default=None, help="Override docker.gpu_id.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded jobs / docker command without executing.",
    )
    args = parser.parse_args()

    spec = load_experiment_spec(args.config)

    docker_cfg = spec.docker or DockerConfig(
        image=DEFAULTS["docker_image"],
        gpu_id=DEFAULTS["docker_gpu_id"],
        cpus=DEFAULTS["docker_cpus"],
        memory=DEFAULTS["docker_memory"],
        container_workdir=DEFAULTS["docker_container_workdir"],
    )
    if args.docker_gpu is not None:
        docker_cfg.gpu_id = args.docker_gpu

    inner_argv: list[str] = []
    if args.folds:
        inner_argv += ["--folds", *[str(f) for f in args.folds]]
    if args.verbose:
        inner_argv.append("--verbose")
    if args.fail_fast:
        inner_argv.append("--fail-fast")
    if args.dataset:
        inner_argv += ["--dataset", args.dataset]
    if args.dry_run:
        inner_argv.append("--dry-run")

    use_docker = (
        not args.no_docker
        and not running_in_docker()
        and (args.docker or spec.docker is not None)
    )
    if use_docker:
        code = docker_run(
            args.config,
            docker_cfg,
            inner_argv=inner_argv,
            dry_run=args.dry_run,
        )
        sys.exit(code)

    batches, run_timestamp = _resolve_batches(spec, dataset_override=args.dataset)
    if spec.campaign is None and args.dataset and batches:
        batches[0].dataset = args.dataset
    batches = _apply_cli_overrides(
        batches,
        dataset=args.dataset if spec.campaign is not None else None,
        folds=args.folds,
    )
    if not batches:
        print("No jobs matched the requested filters.")
        sys.exit(1)

    started_at = utc_now_iso()
    results_dir = batches[0].run.results_dir if batches else DEFAULTS["results_dir"]
    batch_results: list[BatchRunResult] = []
    all_failed: list[tuple[str, int, int, str]] = []
    for batch in batches:
        result = _run_batch(
            batch,
            verbose=args.verbose,
            fail_fast=args.fail_fast,
            dry_run=args.dry_run,
        )
        batch_results.append(result)
        for mode, fold, code in result.failed:
            all_failed.append((result.experiment_id, mode, fold, code))
        if result.failed and args.fail_fast:
            sys.exit(result.failed[0][2])

    if not args.dry_run and batch_results:
        summary_cfg = resolve_summary_config(spec, len(batch_results))
        manifest = ExperimentManifest(
            event_description=resolve_event_description(spec),
            timestamp=run_timestamp,
            config_path=str(Path(args.config).as_posix()),
            started_at=started_at,
            finished_at=utc_now_iso(),
            docker=docker_to_dict(spec.docker),
            summary={
                "layout": summary_cfg.resolve_layout(len(batch_results)),
                "metrics": list(summary_cfg.metrics),
                "datasets": summary_cfg.datasets,
            },
            runs=[
                build_run_entry(
                    result.batch,
                    result.experiment_id,
                    status=result.status,
                    results_dir=results_dir,
                )
                for result in batch_results
            ],
        )
        manifest_path = write_manifest(manifest, results_dir)
        print(f"\nManifest: {manifest_path.resolve()}")

    if all_failed:
        print(f"\n{len(all_failed)} run(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
