"""Static lexical difficulty signals for curriculum ablations."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def tfidf_difficulty(X) -> np.ndarray:
    """L2 norm of each sparse TF-IDF row (higher = lexically harder)."""
    X = sparse.csr_matrix(X)
    norms = np.asarray(sparse.linalg.norm(X, axis=1)).ravel().astype(np.float64)
    if norms.max() == norms.min():
        return np.zeros_like(norms)
    return (norms - norms.min()) / (norms.max() - norms.min())
