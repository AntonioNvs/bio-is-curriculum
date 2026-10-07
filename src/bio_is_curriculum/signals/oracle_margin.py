"""Oracle-margin difficulty signal for Bengio-style curriculum (ICML 2009 §4.2).

Standalone OOF logistic regression on TF-IDF features — no BIOIS dependency.
Higher margin = easier example.
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import unique_labels


def fix_proba_columns_if_necessary(
    probas: np.ndarray,
    columns_diff: set[int],
    max_y_train: int,
) -> np.ndarray:
    """Align fold-local LR probas to the global label space."""
    probas = np.asarray(probas)
    n_instances = probas.shape[0]
    if not columns_diff:
        return probas
    for c in sorted(columns_diff):
        if c == 0:
            probas = np.c_[np.zeros(n_instances), probas]
        elif c == max_y_train:
            probas = np.c_[probas, np.zeros(n_instances)]
        else:
            probas = np.c_[probas[:, :c], np.zeros(n_instances), probas[:, c:]]
    return probas


def oof_lr_probas(
    X,
    y,
    *,
    random_state: int = 0,
    n_splits: int = 5,
) -> np.ndarray:
    """Out-of-fold class probabilities from stratified k-fold LR on TF-IDF."""
    y = np.asarray(y)
    nrows = X.shape[0]
    classes = unique_labels(y)
    ncolumns = len(classes)
    proba_everyone = np.zeros((nrows, ncolumns))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_index, val_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_val = X[val_index]

        classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        classifier.fit(X_train, y_train)

        probas = classifier.predict_proba(X_val)
        columns_diff = set(y) - set(y_train)
        if columns_diff:
            probas = fix_proba_columns_if_necessary(
                probas, columns_diff, int(max(y_train))
            )
        proba_everyone[val_index] = copy.copy(probas)

    return proba_everyone


def multiclass_margins(probas: np.ndarray, y) -> np.ndarray:
    """Per-sample margin P(y_i) - max_{c≠y_i} P(c); higher = easier."""
    probas = np.asarray(probas, dtype=np.float64)
    y_arr = np.asarray(y).astype(int)
    n = len(y_arr)
    margins = np.empty(n, dtype=np.float64)
    for i in range(n):
        yi = y_arr[i]
        p_true = probas[i, yi]
        others = np.delete(probas[i], yi)
        p_best_other = float(np.max(others)) if others.size else 0.0
        margins[i] = p_true - p_best_other
    return margins


def easy_indices_by_margin(
    margin: np.ndarray,
    y,
    easy_fraction: float,
    *,
    use_global: bool = True,
) -> np.ndarray:
    """Indices of the easiest ``easy_fraction`` of training examples."""
    margin = np.asarray(margin, dtype=np.float64)
    y = np.asarray(y)
    n = len(margin)
    if n == 0:
        return np.empty(0, dtype=int)

    easy_fraction = float(np.clip(easy_fraction, 0.0, 1.0))
    if easy_fraction >= 1.0:
        return np.arange(n, dtype=int)

    if use_global:
        k = max(1, int(np.ceil(easy_fraction * n)))
        order = np.argsort(-margin, kind="stable")
        return np.sort(order[:k])

    from bio_is_curriculum.curriculum.class_balance import per_class_high_quantile_mask

    mask = per_class_high_quantile_mask(margin, y, easy_fraction)
    return np.flatnonzero(mask)
