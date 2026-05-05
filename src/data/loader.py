import io
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


class DatasetLoader:
    """
    Loads text data, labels, and pre-defined splits for text classification datasets.
    Assumes a directory structure where each dataset has 'texts.txt', 'score.txt', 
    and a 'splits' folder containing .pkl split files.
    """
    def __init__(self, data_dir: str, dataset_name: str):
        self.dataset_path = os.path.join(data_dir, dataset_name)
    
    def load_texts_and_scores(self):
        """
        Loads the texts and scores from the dataset directory.
        Returns:
            texts (list): A list of document texts.
            scores (list): A list of labels for the documents.
        """
        texts_path = os.path.join(self.dataset_path, 'texts.txt')
        score_path = os.path.join(self.dataset_path, 'score.txt')
        
        if not os.path.exists(texts_path) or not os.path.exists(score_path):
            raise FileNotFoundError(f"Missing texts.txt or score.txt in {self.dataset_path}")

        with io.open(texts_path, 'rt', newline='\n', encoding='utf-8', errors='ignore') as f:
            texts = [line.rstrip('\n').strip() for line in f.readlines()]

        with io.open(score_path, 'rt', newline='\n', encoding='utf-8', errors='ignore') as f:
            scores = [line.rstrip('\n').strip() for line in f.readlines()]
            try:
                # Attempt to safely convert numeric discrete scores to integers
                scores = [int(s) if s.isdigit() or (s.startswith('-') and s[1:].isdigit()) else s for s in scores]
            except ValueError:
                pass
                
        return texts, scores
    
    def load_splits(self, n_splits=10, with_val=False):
        """
        Loads split indices from the standard .pkl files (e.g., split_10.pkl).
        These `.pkl` files contain a DataFrame with fold structure.
        """
        suffix = f"_{n_splits}_with_val.pkl" if with_val else f"_{n_splits}.pkl"
        pkl_path = os.path.join(self.dataset_path, 'splits', f"split{suffix}")
        
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Split file not found: {pkl_path}")
            
        df_splits = pd.read_pickle(pkl_path)
        return df_splits

    def load_texts_fold(self, fold: int, n_splits: int = 10, with_val: bool = False):
        """Load raw texts and labels for a given fold, aligned with load_tfidf_fold.

        The split .pkl stores train_idxs/test_idxs that match the rows of the
        pre-built TF-IDF files (same upstream generation pipeline), so index i
        from BIOIS/curriculum maps to texts_train[i] for the same instance.

        Training data is shuffled using StratifiedShuffleSplit to avoid issues
        with class-sequential ordering in the original split files.

        Args:
            fold: Fold index to load.
            n_splits: Total number of folds (used to locate the split file).
            with_val: If True, carves out a stratified 10% validation set from
                training data and returns it as a 6-tuple.

        Returns:
            without with_val: (texts_train, y_train, texts_test, y_test)
            with    with_val: (texts_train, y_train, texts_val, y_val, texts_test, y_test)
        """
        texts, scores = self.load_texts_and_scores()
        df = self.load_splits(n_splits=n_splits)
        row = df[df["fold_id"] == fold].iloc[0]
        train_idx = list(row["train_idxs"])
        test_idx = list(row["test_idxs"])

        train_scores = [scores[i] for i in train_idx]
        test_scores = [scores[i] for i in test_idx]

        le = LabelEncoder().fit(train_scores)
        y_train_full = le.transform(train_scores)
        y_test = le.transform(test_scores)

        texts_train_full = [texts[i] for i in train_idx]
        texts_test = [texts[i] for i in test_idx]

        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=2026)
        for sss_train_idx, sss_val_idx in sss.split(texts_train_full, y_train_full):
            continue

        texts_train = [texts_train_full[i] for i in sss_train_idx]
        y_train = y_train_full[sss_train_idx]

        if with_val:
            texts_val = [texts_train_full[i] for i in sss_val_idx]
            y_val = y_train_full[sss_val_idx]
            return texts_train, y_train, texts_val, y_val, texts_test, y_test

        return texts_train, y_train, texts_test, y_test

    def load_tfidf_fold(self, fold: int, with_val: bool = False):
        """Load the prebuilt per-fold TF-IDF matrices and labels.

        Mirrors the upstream bio-is loader (``utils/general.py::get_data``):
        reads ``tfidf/train{fold}.gz`` and ``tfidf/test{fold}.gz`` (svmlight
        format), aligns feature dimensionality between train and test, and
        encodes labels to a 0..n-1 contiguous integer range using a
        ``LabelEncoder`` fitted on the training labels.

        Training data is shuffled using StratifiedShuffleSplit to avoid issues
        with class-sequential ordering in the original split files.

        Args:
            fold: Fold index to load.
            with_val: If True, carves out a stratified 10% validation set from
                training data and returns it as a 6-tuple.

        Returns:
            without with_val: (X_train, y_train, X_test, y_test)
            with    with_val: (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        tfidf_dir = os.path.join(self.dataset_path, "tfidf")
        train_path = os.path.join(tfidf_dir, f"train{fold}.gz")
        test_path = os.path.join(tfidf_dir, f"test{fold}.gz")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Missing prebuilt TF-IDF files for fold {fold} in {tfidf_dir}"
            )

        X_train_full, y_train_raw = load_svmlight_file(train_path, dtype=np.float64)
        X_test, y_test_raw = load_svmlight_file(test_path, dtype=np.float64)

        if X_train_full.shape[1] != X_test.shape[1]:
            n_features = max(X_train_full.shape[1], X_test.shape[1])
            X_train_full, y_train_raw = load_svmlight_file(
                train_path, dtype=np.float64, n_features=n_features
            )
            X_test, y_test_raw = load_svmlight_file(
                test_path, dtype=np.float64, n_features=n_features
            )

        le = LabelEncoder().fit(y_train_raw)
        y_train_full = le.transform(y_train_raw)
        y_test = le.transform(y_test_raw)

        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=2018)
        for sss_train_idx, sss_val_idx in sss.split(X_train_full, y_train_full):
            continue

        X_train = X_train_full[sss_train_idx]
        y_train = y_train_full[sss_train_idx]

        if with_val:
            X_val = X_train_full[sss_val_idx]
            y_val = y_train_full[sss_val_idx]
            return X_train, y_train, X_val, y_val, X_test, y_test

        return X_train, y_train, X_test, y_test
