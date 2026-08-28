"""Base classes for literature curriculum-learning baselines."""

from abc import ABCMeta, abstractmethod

from bio_is_curriculum.curriculum.methods.biois_discrete import BIOISDiscreteCurriculum


class BaselineBase(BIOISDiscreteCurriculum, metaclass=ABCMeta):
    """Phased CL baselines reuse the BIOIS discrete orchestrator."""

    INDEX: int = -1
    NAME: str = "abstract"
    REFERENCE: str = ""
    TRAINER_KIND: str = "phased"


class DynamicBaselineBase(metaclass=ABCMeta):
    """Dynamic CL baselines with per-epoch reordering (e.g. SPDCL)."""

    INDEX: int = -1
    NAME: str = "abstract"
    REFERENCE: str = ""
    TRAINER_KIND: str = "dynamic"

    model_: object = None
    history_: list = []

    @abstractmethod
    def fit(self, selector, X, y, **kwargs):
        ...
