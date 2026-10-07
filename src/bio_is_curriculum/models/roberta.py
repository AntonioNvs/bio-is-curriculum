"""Deprecated alias for ModernBertModel (legacy RoBERTa backend name)."""

from __future__ import annotations

import warnings

from bio_is_curriculum.models.modernbert import ModernBertModel

__all__ = ["RobertaModel"]


class RobertaModel(ModernBertModel):
    """Deprecated: use :class:`ModernBertModel` with ``model: modernbert``."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "RobertaModel is deprecated; use ModernBertModel with model='modernbert'.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
