"""Campaign matrix expansion into batch experiment configs."""

from __future__ import annotations

import copy
import itertools
import re
from datetime import datetime
from typing import Any

from bio_is_curriculum.config.loader import batch_from_yaml_dict
from bio_is_curriculum.config.schema import (
    BatchExperimentConfig,
    CampaignJobSpec,
    CampaignSpec,
)

_MATRIX_KEY_ALIASES = {
    "curriculum.method": "method",
    "curriculum.loss_scheme": "loss_scheme",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _set_dotted_key(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = data
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _matrix_combos(matrix: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not matrix:
        return [{}]
    keys = list(matrix.keys())
    values = [matrix[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _template_vars(
    *,
    dataset: str,
    n_splits: int,
    timestamp: str,
    matrix_combo: dict[str, Any],
) -> dict[str, str]:
    vars_: dict[str, str] = {
        "dataset": dataset,
        "n_splits": str(n_splits),
        "timestamp": timestamp,
    }
    for key, value in matrix_combo.items():
        short = _MATRIX_KEY_ALIASES.get(key, key.split(".")[-1])
        vars_[short] = str(value)
        vars_[key.replace(".", "_")] = str(value)
    return vars_


def _render_template(template: str, variables: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            raise KeyError(f"Unknown experiment_id template variable: {key}")
        return variables[key]

    return re.sub(r"\{(\w+)\}", repl, template)


def _job_yaml_for_combo(
    campaign: CampaignSpec,
    job: CampaignJobSpec,
    *,
    dataset: str,
    n_splits: int,
    timestamp: str,
    matrix_combo: dict[str, Any],
) -> dict[str, Any]:
    yaml_cfg = _deep_merge(campaign.defaults, {})
    yaml_cfg["dataset"] = dataset
    yaml_cfg["n_splits"] = n_splits
    yaml_cfg["modes"] = list(job.modes)

    if job.folds is not None:
        yaml_cfg["folds"] = list(job.folds)

    for key, value in matrix_combo.items():
        _set_dotted_key(yaml_cfg, key, value)

    if job.experiment_id:
        variables = _template_vars(
            dataset=dataset,
            n_splits=n_splits,
            timestamp=timestamp,
            matrix_combo=matrix_combo,
        )
        yaml_cfg["experiment_id"] = _render_template(job.experiment_id, variables)

    return yaml_cfg


def resolve_campaign_timestamp(
    campaign: CampaignSpec,
    *,
    timestamp: str | None = None,
) -> str:
    """Return the shared timestamp string for a campaign invocation."""
    if timestamp is not None:
        return timestamp
    if campaign.timestamp == "auto":
        return datetime.now().strftime("%Y%m%d-%H%M%S")
    return campaign.timestamp


def expand_campaign(
    campaign: CampaignSpec,
    *,
    timestamp: str | None = None,
) -> list[BatchExperimentConfig]:
    """Expand campaign jobs into a flat list of batch configs."""
    if not campaign.datasets:
        raise ValueError("Campaign must define at least one dataset under campaign.datasets")

    run_timestamp = resolve_campaign_timestamp(campaign, timestamp=timestamp)

    default_n_splits = int(campaign.defaults.get("n_splits", 10))
    batches: list[BatchExperimentConfig] = []

    for dataset, ds_cfg in campaign.datasets.items():
        ds_cfg = ds_cfg or {}
        n_splits = int(ds_cfg.get("n_splits", default_n_splits))

        for job in campaign.jobs:
            for matrix_combo in _matrix_combos(job.matrix):
                yaml_cfg = _job_yaml_for_combo(
                    campaign,
                    job,
                    dataset=dataset,
                    n_splits=n_splits,
                    timestamp=run_timestamp,
                    matrix_combo=matrix_combo,
                )
                batches.append(batch_from_yaml_dict(yaml_cfg))

    return batches
