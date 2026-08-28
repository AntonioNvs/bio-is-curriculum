"""Oversampling minimo por classe para estabilizar StratifiedKFold / treino LR."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class UpsampleStats:
    """Estatisticas de um passe de oversampling.

    `dup_row_idx` traz os indices (na ordem `X` original) que foram
    duplicados — exposto para que codigos chamadores possam estender, com
    o MESMO conjunto de duplicacoes, arrays paralelos ao X (e.g., sinais
    do BIOIS no curriculum is_cl).
    """

    n_before: int
    n_after: int
    dup_row_idx: np.ndarray  # shape (n_added,)

    @property
    def n_added(self) -> int:
        return self.n_after - self.n_before


def upsample_min_per_class(
    X,
    y,
    *,
    min_count: int = 5,
    random_state: int | None = None,
    texts: list[str] | None = None,
) -> tuple[object, np.ndarray, UpsampleStats, list[str] | None]:
    """Garante pelo menos ``min_count`` exemplos por rotulo presente em ``y``.

    Aceita TF-IDF (sparse/densa) ou lista de linhas (ex.: strings para RoBERTa).
    Com ``texts`` paralelo ao TF-IDF, duplica matriz e textos mantendo indice-alinhamento.

    Returns:
        (X_new, y_new, stats, texts_new); ``texts_new`` e None quando ``texts`` e None.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.ravel()

    plain_list_rows = isinstance(X, list)

    if plain_list_rows:
        n_before = len(X)
    else:
        n_before = int(X.shape[0])

    if texts is not None and len(texts) != n_before:
        raise ValueError(f"len(texts)={len(texts)} != n_samples={n_before}")
    if len(y) != n_before:
        raise ValueError(f"len(y)={len(y)} != n_samples={n_before}")

    rng = np.random.default_rng(random_state)
    dup_row_idx: list[int] = []

    for c in np.unique(y):
        cls_idx = np.flatnonzero(y == c)
        k = cls_idx.size
        if k == 0:
            continue
        if k < min_count:
            need = min_count - k
            dup_row_idx.extend(rng.choice(cls_idx, size=need, replace=True).tolist())

    if not dup_row_idx:
        stats = UpsampleStats(
            n_before=n_before,
            n_after=n_before,
            dup_row_idx=np.empty(0, dtype=int),
        )
        if plain_list_rows:
            return list(X), np.asarray(y, dtype=y.dtype), stats, texts
        return X, np.asarray(y, dtype=y.dtype), stats, texts

    dup_arr = np.array(dup_row_idx, dtype=int)
    y_extra = y[dup_arr]
    y_new = np.concatenate([y, y_extra])
    yt = np.asarray(y_new, dtype=y.dtype, copy=False)

    if plain_list_rows:
        X_new_list = list(X)
        X_new_list.extend(X[i] for i in dup_row_idx)
        if texts is None:
            texts_new = None
        else:
            texts_new = list(texts)
            texts_new.extend(texts[i] for i in dup_row_idx)
        stats = UpsampleStats(
            n_before=n_before, n_after=len(X_new_list), dup_row_idx=dup_arr,
        )
        return X_new_list, yt, stats, texts_new

    X_extra = X[dup_arr]
    if sparse.issparse(X):
        X_new = sparse.vstack([X, X_extra], format="csr")
    else:
        X_new = np.vstack([np.asarray(X), np.asarray(X_extra)])

    if texts is not None:
        texts_new = list(texts)
        texts_new.extend(texts[i] for i in dup_row_idx)
    else:
        texts_new = None

    stats = UpsampleStats(
        n_before=n_before, n_after=int(X_new.shape[0]), dup_row_idx=dup_arr,
    )
    return X_new, yt, stats, texts_new
