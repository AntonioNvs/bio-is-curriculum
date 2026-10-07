"""Abstract model interface for curriculum training."""

from abc import ABCMeta, abstractmethod


class CurriculumModel(metaclass=ABCMeta):
    """Minimal interface required by curriculum orchestrators."""

    @abstractmethod
    def fit_stage(
        self,
        X,
        y,
        sample_weight=None,
        X_val=None,
        y_val=None,
        balanced_sampling: bool = False,
    ):
        """Continue training for one curriculum phase."""
        ...

    @abstractmethod
    def predict(self, X):
        ...

    @abstractmethod
    def predict_proba(self, X):
        ...

    @property
    @abstractmethod
    def n_iter(self) -> int:
        ...

    def get_training_stats(self) -> dict:
        return {
            "avg_seq_len": float("nan"),
            "compute_proxy": float("nan"),
            "best_val_macro_f1": float("nan"),
            "best_val_epoch": float("nan"),
            "steps_to_best_val": float("nan"),
        }

    def set_phase(self, phase_name: str) -> None:
        pass

    def extract_hidden_states(self, texts: list[str]) -> list:
        """Return last-layer token hidden states per sample (transformer PLM only)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support hidden-state extraction."
        )

    def fit_epoch(
        self,
        texts: list[str],
        y,
        sample_weight=None,
        X_val=None,
        y_val=None,
        epoch_seed: int | None = None,
    ) -> None:
        """Train for a single epoch (dynamic curricula). Defaults to one fit_stage epoch."""
        prev = getattr(self, "epochs_per_stage", 1)
        if hasattr(self, "epochs_per_stage"):
            self.epochs_per_stage = 1
        self.fit_stage(texts, y, sample_weight=sample_weight, X_val=X_val, y_val=y_val)
        if hasattr(self, "epochs_per_stage"):
            self.epochs_per_stage = prev
