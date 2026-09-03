"""Experiment configuration dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from bio_is_curriculum.config.defaults import DEFAULTS, SCHEMA_VERSION


def _field_names(cls) -> set[str]:
    return {f.name for f in fields(cls)}


@dataclass
class ExperimentConfig:
    """Full configuration for a single fold run."""

    schema_version: str = SCHEMA_VERSION
    dataset: str = ""
    data_dir: str = DEFAULTS["data_dir"]
    fold: int = 0
    n_splits: int = 10
    mode: str = "is_cl"
    baseline: int | None = None
    beta: float = DEFAULTS["beta"]
    theta: float = DEFAULTS["theta"]
    random_state: int = DEFAULTS["random_state"]
    curriculum_method: str = DEFAULTS["curriculum_method"]
    curriculum_beta: float = DEFAULTS["curriculum_beta"]
    curriculum_q: tuple[float, float, float] = (
        DEFAULTS["curriculum_q_low"],
        DEFAULTS["curriculum_q_mid"],
        DEFAULTS["curriculum_q_high"],
    )
    curriculum_n_steps: int = DEFAULTS["curriculum_n_steps"]
    curriculum_alpha_decay: float = DEFAULTS["curriculum_alpha_decay"]
    curriculum_soft_lambda_init: float = DEFAULTS["curriculum_soft_lambda_init"]
    curriculum_soft_lambda_growth: float = DEFAULTS["curriculum_soft_lambda_growth"]
    curriculum_soft_lambda_max: float = DEFAULTS["curriculum_soft_lambda_max"]
    curriculum_soft_min_weight: float = DEFAULTS["curriculum_soft_min_weight"]
    curriculum_soft_stability_tol: float = DEFAULTS["curriculum_soft_stability_tol"]
    curriculum_soft_saturation_patience: int = DEFAULTS["curriculum_soft_saturation_patience"]
    curriculum_soft_max_effective_steps: int = DEFAULTS["curriculum_soft_max_effective_steps"]
    curriculum_loss_scheme: str = DEFAULTS["curriculum_loss_scheme"]
    curriculum_lambda_init: float = DEFAULTS["curriculum_lambda_init"]
    curriculum_lambda_step: float = DEFAULTS["curriculum_lambda_step"]
    curriculum_lambda_mult: float = DEFAULTS["curriculum_lambda_mult"]
    curriculum_lambda_max: float | None = DEFAULTS["curriculum_lambda_max"]
    curriculum_lambda2: float | None = DEFAULTS["curriculum_lambda2"]
    curriculum_loss_prior_reliability: bool = DEFAULTS["curriculum_loss_prior_reliability"]
    curriculum_min_weight: float = DEFAULTS["curriculum_min_weight"]
    curriculum_loss_recompute_every: int = DEFAULTS["curriculum_loss_recompute_every"]
    spdcl_n_bins: int = DEFAULTS["spdcl_n_bins"]
    spdcl_curriculum_epochs: int | None = DEFAULTS["spdcl_curriculum_epochs"]
    spdcl_anneal_epochs: int = DEFAULTS["spdcl_anneal_epochs"]
    spdcl_norm_subsample: int | None = DEFAULTS["spdcl_norm_subsample"]
    b1_easy_fraction: float = DEFAULTS["b1_easy_fraction"]
    b1_use_global_quantile: bool = DEFAULTS["b1_use_global_quantile"]
    model: str = DEFAULTS["model"]
    hf_model: str = DEFAULTS["hf_model"]
    train_fraction: float = DEFAULTS["train_fraction"]
    epochs: int = DEFAULTS["epochs"]
    epochs_per_phase: int = DEFAULTS["epochs_per_phase"]
    batch_size: int = DEFAULTS["batch_size"]
    eval_batch_size: int = DEFAULTS["eval_batch_size"]
    max_length: int = DEFAULTS["max_length"]
    lr: float = DEFAULTS["lr"]
    weight_decay: float = DEFAULTS["weight_decay"]
    warmup_ratio: float = DEFAULTS["warmup_ratio"]
    imbalance_method: str = DEFAULTS["imbalance_method"]
    effective_num_beta: float = DEFAULTS["effective_num_beta"]
    dist_bal_tau: float = DEFAULTS["dist_bal_tau"]
    dist_bal_logit_bias: float = DEFAULTS["dist_bal_logit_bias"]
    aug_target_min_count: int = DEFAULTS["aug_target_min_count"]
    aug_ratio: float = DEFAULTS["aug_ratio"]
    aug_random_swap: float = DEFAULTS["aug_random_swap"]
    aug_random_delete: float = DEFAULTS["aug_random_delete"]
    class_balanced_loss: bool | None = DEFAULTS["class_balanced_loss"]
    hard_slice_quantile: float = DEFAULTS["hard_slice_quantile"]
    cuda_device_id: int = DEFAULTS["cuda_device_id"]
    results_dir: str = DEFAULTS["results_dir"]
    experiment_id: str | None = None
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["curriculum_q"] = list(self.curriculum_q)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        known = _field_names(cls)
        kwargs = {k: v for k, v in data.items() if k in known}
        if "curriculum_q" in kwargs and isinstance(kwargs["curriculum_q"], list):
            kwargs["curriculum_q"] = tuple(kwargs["curriculum_q"])
        return cls(**kwargs)

    def resolve_mode(self) -> str:
        if self.baseline is not None:
            return f"b{self.baseline}"
        return self.mode

    def resolve_curriculum_method(self) -> str:
        if self.curriculum_method:
            return self.curriculum_method
        if self.mode == "is_continuous_cl":
            return "spcl_soft"
        return "biois_discrete"


@dataclass
class BatchExperimentConfig:
    """Configuration for multi-fold YAML experiments."""

    dataset: str = ""
    n_splits: int = 10
    modes: list[str] = field(default_factory=lambda: ["raw", "is", "cl", "is_cl"])
    folds: list[int] | None = None
    experiment_id: str | None = None
    run: ExperimentConfig = field(default_factory=ExperimentConfig)


@dataclass
class DockerConfig:
    """Docker execution settings for host-side experiment launches."""

    image: str = DEFAULTS["docker_image"]
    gpu_id: int = DEFAULTS["docker_gpu_id"]
    cpus: int = DEFAULTS["docker_cpus"]
    memory: str = DEFAULTS["docker_memory"]
    host_project_dir: str | None = None
    container_workdir: str = DEFAULTS["docker_container_workdir"]

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DockerConfig | None:
        if not data:
            return None
        return cls(
            image=data.get("image", DEFAULTS["docker_image"]),
            gpu_id=int(data.get("gpu_id", DEFAULTS["docker_gpu_id"])),
            cpus=int(data.get("cpus", DEFAULTS["docker_cpus"])),
            memory=str(data.get("memory", DEFAULTS["docker_memory"])),
            host_project_dir=data.get("host_project_dir"),
            container_workdir=str(
                data.get("container_workdir", DEFAULTS["docker_container_workdir"])
            ),
        )


DEFAULT_SUMMARY_METRICS = [
    "macro_f1",
    "micro_f1",
    "f1_weighted",
    "accuracy",
    "hard_slice_macro_f1",
    "train_time_s",
    "total_time",
]


@dataclass
class SummaryConfig:
    """Excel/CSV export settings copied into experiment manifests."""

    layout: str = "auto"
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_SUMMARY_METRICS))
    datasets: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SummaryConfig | None:
        if not data:
            return None
        metrics = data.get("metrics")
        return cls(
            layout=str(data.get("layout", "auto")),
            metrics=list(metrics) if metrics else list(DEFAULT_SUMMARY_METRICS),
            datasets=list(data["datasets"]) if data.get("datasets") else None,
        )

    def resolve_layout(self, num_runs: int) -> str:
        if self.layout == "auto":
            return "compare_by_dataset" if num_runs > 1 else "long_table"
        return self.layout

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "metrics": list(self.metrics),
            "datasets": self.datasets,
        }


@dataclass
class CampaignJobSpec:
    """One job entry in a campaign matrix."""

    modes: list[str] = field(default_factory=list)
    matrix: dict[str, list[Any]] = field(default_factory=dict)
    experiment_id: str | None = None
    folds: list[int] | None = None


@dataclass
class CampaignSpec:
    """Multi-dataset campaign with matrix job expansion."""

    name: str | None = None
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    jobs: list[CampaignJobSpec] = field(default_factory=list)
    timestamp: str = "auto"
    summary: SummaryConfig | None = None


@dataclass
class ExperimentSpec:
    """Top-level experiment file: simple batch or campaign."""

    config_path: str = ""
    docker: DockerConfig | None = None
    campaign: CampaignSpec | None = None
    batch: BatchExperimentConfig | None = None
    summary: SummaryConfig | None = None
