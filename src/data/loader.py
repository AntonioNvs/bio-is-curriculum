import os
import pandas as pd

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
