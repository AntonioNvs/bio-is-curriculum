"""Data preprocessing helpers extracted from the CLI orchestrator."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from bio_is_curriculum.data.rare_class_upsampling import upsample_min_per_class

BIOIS_STRATKFOLD_SPLITS = 5
VAL_SPLIT_SEED = 2018


def print_oversampling(stage: str, stats) -> None:
    if stats.n_added == 0:
        print(f"  Oversampling ({stage}): no changes (n={stats.n_before})")
    else:
        print(
            f"  Oversampling ({stage}): {stats.n_before} -> {stats.n_after} "
            f"(+{stats.n_added} instances)"
        )


def split_train_val(
    X_train_raw,
    y_train_raw,
    texts_train_raw: list[str] | None,
    y_texts_raw,
    *,
    random_state: int,
):
    """Stratified 90/10 train/val split with rare-class stabilization."""
    X_train_raw, y_train_raw, st_raw, texts_train_raw = upsample_min_per_class(
        X_train_raw,
        y_train_raw,
        min_count=2,
        random_state=random_state,
        texts=texts_train_raw,
    )
    if y_texts_raw is not None:
        y_texts_raw = y_train_raw
    print_oversampling("pre-split (rare class stabilization)", st_raw)

    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=VAL_SPLIT_SEED)
    for train_idx, val_idx in sss.split(X_train_raw, y_train_raw):
        pass

    X_train = X_train_raw[train_idx]
    y_train = y_train_raw[train_idx]
    X_val = X_train_raw[val_idx]
    y_val = y_train_raw[val_idx]

    texts_train = texts_val = None
    y_texts_train = y_texts_val = None
    if texts_train_raw is not None:
        texts_train = [texts_train_raw[i] for i in train_idx]
        texts_val = [texts_train_raw[i] for i in val_idx]
        y_texts_train = y_texts_raw[train_idx]
        y_texts_val = y_texts_raw[val_idx]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "texts_train": texts_train,
        "texts_val": texts_val,
        "y_texts_train": y_texts_train,
        "y_texts_val": y_texts_val,
    }


def maybe_upsample_for_biois(X_train, y_train, texts_train, random_state: int):
    if texts_train is not None:
        X_train, y_train, stats, texts_train = upsample_min_per_class(
            X_train,
            y_train,
            min_count=BIOIS_STRATKFOLD_SPLITS,
            random_state=random_state,
            texts=texts_train,
        )
        print_oversampling("pre-IS (BIOIS view)", stats)
        return X_train, y_train, texts_train, np.asarray(y_train)
    X_train, y_train, stats, _ = upsample_min_per_class(
        X_train,
        y_train,
        min_count=BIOIS_STRATKFOLD_SPLITS,
        random_state=random_state,
    )
    print_oversampling("pre-IS (BIOIS view)", stats)
    return X_train, y_train, texts_train, None


def print_class_distribution(y_train) -> None:
    print(f"  Train classes (TF-IDF): {Counter(np.asarray(y_train).tolist())}")
