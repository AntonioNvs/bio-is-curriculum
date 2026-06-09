"""Metodos de curriculum learning."""
from .biois_discrete import BIOISDiscreteCurriculum
from .registry import (
    ALIASES,
    REGISTRY,
    build_curriculum_kwargs,
    get_curriculum_method,
    resolve_method_id,
)
from .spcl_loss import SPCLLossCurriculum
from .spcl_soft import SPCLSoftCurriculum

__all__ = [
    "ALIASES",
    "REGISTRY",
    "BIOISDiscreteCurriculum",
    "SPCLSoftCurriculum",
    "SPCLLossCurriculum",
    "build_curriculum_kwargs",
    "get_curriculum_method",
    "resolve_method_id",
]
