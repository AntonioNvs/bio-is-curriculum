"""Load experiment configuration from YAML and CLI overrides."""

from __future__ import annotations

from typing import Any

from bio_is_curriculum.config.defaults import DEFAULTS
from bio_is_curriculum.config.schema import (
    BatchExperimentConfig,
    CampaignJobSpec,
    CampaignSpec,
    DockerConfig,
    ExperimentConfig,
    ExperimentSpec,
    SummaryConfig,
)


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_yaml_to_experiment_config(yaml_cfg: dict) -> ExperimentConfig:
    """Merge YAML experiment file into ExperimentConfig."""
    data: dict[str, Any] = {}

    for key in (
        "dataset", "n_splits", "model", "hf_model", "results_dir", "data_dir",
        "random_state", "hard_slice_quantile", "experiment_id", "cuda_device_id",
    ):
        if key in yaml_cfg:
            data[key] = yaml_cfg[key]

    is_cfg = yaml_cfg.get("instance_selection", {})
    for key in ("beta", "theta"):
        if key in is_cfg:
            data[key] = is_cfg[key]

    t_cfg = yaml_cfg.get("training", {})
    training_keys = (
        "epochs", "epochs_per_phase", "batch_size", "eval_batch_size",
        "max_length", "lr", "weight_decay", "warmup_ratio", "imbalance_method",
        "effective_num_beta", "dist_bal_tau", "dist_bal_logit_bias",
        "aug_target_min_count", "aug_ratio", "aug_random_swap", "aug_random_delete",
        "cuda_device_id",
    )
    for key in training_keys:
        if key in t_cfg:
            data[key] = t_cfg[key]
    if "imbalance_method" not in t_cfg and "class_balanced_loss" in t_cfg:
        data["imbalance_method"] = (
            "inverse_freq_ce" if bool(t_cfg["class_balanced_loss"]) else "none"
        )

    c_cfg = yaml_cfg.get("curriculum", {})
    curriculum_map = {
        "method": "curriculum_method",
        "beta": "curriculum_beta",
        "n_steps": "curriculum_n_steps",
        "alpha_decay": "curriculum_alpha_decay",
        "soft_lambda_init": "curriculum_soft_lambda_init",
        "soft_lambda_growth": "curriculum_soft_lambda_growth",
        "soft_lambda_max": "curriculum_soft_lambda_max",
        "soft_min_weight": "curriculum_soft_min_weight",
        "soft_stability_tol": "curriculum_soft_stability_tol",
        "soft_saturation_patience": "curriculum_soft_saturation_patience",
        "soft_max_effective_steps": "curriculum_soft_max_effective_steps",
        "loss_scheme": "curriculum_loss_scheme",
        "lambda_init": "curriculum_lambda_init",
        "lambda_step": "curriculum_lambda_step",
        "lambda_mult": "curriculum_lambda_mult",
        "lambda_max": "curriculum_lambda_max",
        "lambda2": "curriculum_lambda2",
        "loss_prior_reliability": "curriculum_loss_prior_reliability",
        "min_weight": "curriculum_min_weight",
        "loss_recompute_every": "curriculum_loss_recompute_every",
    }
    for src, dst in curriculum_map.items():
        if src in c_cfg:
            data[dst] = c_cfg[src]
    if all(k in c_cfg for k in ("q_low", "q_mid", "q_high")):
        data["curriculum_q"] = (c_cfg["q_low"], c_cfg["q_mid"], c_cfg["q_high"])

    b_cfg = yaml_cfg.get("baseline", {})
    for key in (
        "spdcl_n_bins",
        "spdcl_curriculum_epochs",
        "spdcl_anneal_epochs",
        "spdcl_norm_subsample",
        "b1_easy_fraction",
        "b1_use_global_quantile",
    ):
        if key in b_cfg:
            data[key] = b_cfg[key]

    merged = {**DEFAULTS, **data}
    return ExperimentConfig.from_dict(merged)


def batch_from_yaml_dict(raw: dict) -> BatchExperimentConfig:
    """Build BatchExperimentConfig from a flat or partially nested YAML dict."""
    run_cfg = merge_yaml_to_experiment_config(raw)
    return BatchExperimentConfig(
        dataset=raw.get("dataset", run_cfg.dataset),
        n_splits=raw.get("n_splits", run_cfg.n_splits),
        modes=raw.get("modes", ["raw", "is", "cl", "is_cl"]),
        folds=raw.get("folds"),
        experiment_id=raw.get("experiment_id"),
        run=run_cfg,
    )


def _parse_campaign_datasets(raw: dict) -> dict[str, dict[str, Any]]:
    datasets_raw = raw.get("datasets", {})
    if isinstance(datasets_raw, list):
        return {name: {} for name in datasets_raw}
    if not isinstance(datasets_raw, dict):
        raise ValueError("campaign.datasets must be a mapping or list of names")
    parsed: dict[str, dict[str, Any]] = {}
    for name, cfg in datasets_raw.items():
        parsed[str(name)] = cfg if isinstance(cfg, dict) else {}
    return parsed


def _parse_campaign_jobs(raw_jobs: list) -> list[CampaignJobSpec]:
    jobs: list[CampaignJobSpec] = []
    for entry in raw_jobs:
        if not isinstance(entry, dict):
            raise ValueError("Each campaign job must be a mapping")
        jobs.append(
            CampaignJobSpec(
                modes=list(entry.get("modes", [])),
                matrix=dict(entry.get("matrix", {})),
                experiment_id=entry.get("experiment_id"),
                folds=entry.get("folds"),
            )
        )
    return jobs


def load_experiment_spec(path: str) -> ExperimentSpec:
    """Load simple batch or campaign experiment YAML."""
    raw = _load_yaml(path)
    docker = DockerConfig.from_dict(raw.get("docker"))

    if "campaign" in raw:
        camp_raw = raw["campaign"] or {}
        campaign = CampaignSpec(
            name=camp_raw.get("name"),
            datasets=_parse_campaign_datasets(camp_raw),
            defaults=dict(camp_raw.get("defaults", {})),
            jobs=_parse_campaign_jobs(camp_raw.get("jobs", [])),
            timestamp=str(camp_raw.get("timestamp", "auto")),
            summary=SummaryConfig.from_dict(camp_raw.get("summary")),
        )
        return ExperimentSpec(config_path=path, docker=docker, campaign=campaign)

    return ExperimentSpec(
        config_path=path,
        docker=docker,
        batch=batch_from_yaml_dict(raw),
        summary=SummaryConfig.from_dict(raw.get("summary")),
    )


def load_batch_config(path: str) -> BatchExperimentConfig:
    """Load a simple (non-campaign) YAML file into BatchExperimentConfig."""
    spec = load_experiment_spec(path)
    if spec.batch is None:
        raise ValueError(
            f"{path} is a campaign config; use load_experiment_spec() and expand_campaign()"
        )
    return spec.batch
