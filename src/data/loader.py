import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
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
            
        with open(texts_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()]
            
        with open(score_path, 'r', encoding='utf-8') as f:
            scores = [line.strip() for line in f.readlines()]
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

    def load_tfidf_fold(self, fold: int):
        """Load the prebuilt per-fold TF-IDF matrices and labels.

        Mirrors the upstream bio-is loader (``utils/general.py::get_data``):
        reads ``tfidf/train{fold}.gz`` and ``tfidf/test{fold}.gz`` (svmlight
        format), aligns feature dimensionality between train and test, and
        encodes labels to a 0..n-1 contiguous integer range using a
        ``LabelEncoder`` fitted on the training labels.

        Returns:
            X_train (csr_matrix), y_train (ndarray), X_test (csr_matrix), y_test (ndarray)
        """
        tfidf_dir = os.path.join(self.dataset_path, "tfidf")
        train_path = os.path.join(tfidf_dir, f"train{fold}.gz")
        test_path = os.path.join(tfidf_dir, f"test{fold}.gz")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Missing prebuilt TF-IDF files for fold {fold} in {tfidf_dir}"
            )

        X_train, y_train = load_svmlight_file(train_path, dtype=np.float64)
        X_test, y_test = load_svmlight_file(test_path, dtype=np.float64)

        if X_train.shape[1] != X_test.shape[1]:
            n_features = max(X_train.shape[1], X_test.shape[1])
            X_train, y_train = load_svmlight_file(
                train_path, dtype=np.float64, n_features=n_features
            )
            X_test, y_test = load_svmlight_file(
                test_path, dtype=np.float64, n_features=n_features
            )

        le = LabelEncoder().fit(y_train)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

        return X_train, y_train, X_test, y_test
