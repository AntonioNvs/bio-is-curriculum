"""Literature baseline registry."""

from bio_is_curriculum.baselines.base import BaselineBase, DynamicBaselineBase
from bio_is_curriculum.baselines.b1_bengio2009 import Baseline1
from bio_is_curriculum.baselines.b2_spdcl import Baseline2SPDCL

REGISTRY: dict[int, type] = {
    Baseline1.INDEX: Baseline1,
    Baseline2SPDCL.INDEX: Baseline2SPDCL,
}


def get_baseline(index: int):
    if index not in REGISTRY:
        available = sorted(REGISTRY)
        raise ValueError(
            f"Baseline {index} not found. Available: {available}. "
            "See docs/BASELINES.md."
        )
    return REGISTRY[index]


def baseline_run_id(index: int) -> str:
    return f"b{index}"


__all__ = [
    "BaselineBase",
    "DynamicBaselineBase",
    "Baseline1",
    "Baseline2SPDCL",
    "REGISTRY",
    "get_baseline",
    "baseline_run_id",
]
