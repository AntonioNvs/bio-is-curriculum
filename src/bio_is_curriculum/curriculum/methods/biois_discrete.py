"""BIOIS discrete curriculum: clean -> diverse -> hard using entropy signal."""
from __future__ import annotations

from bio_is_curriculum.curriculum.methods.discrete_base import DiscreteCurriculumBase


class BIOISDiscreteCurriculum(DiscreteCurriculumBase):
    """Discrete curriculum over BIOIS redundancy and entropy signals."""

    REQUIRES_BIOIS = True
    METHOD_ID = "biois_discrete"
