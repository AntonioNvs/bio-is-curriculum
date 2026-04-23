from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TextPreprocessor:
    """
    Handles preprocessing and vectorization of text data. 
    By default, sets up a TF-IDF vectorizer ready to process the texts.
    """
    def __init__(self, max_features=None, lowercase=True, stop_words='english', **kwargs):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=lowercase,
            stop_words=stop_words,
            **kwargs
        )
        
    def fit(self, texts, y=None):
        """Fits the TF-IDF representation on a collection of texts."""
        self.vectorizer.fit(texts)
        return self
        
    def transform(self, texts):
        """Transforms texts into sparse TF-IDF vectors."""
        return self.vectorizer.transform(texts)
        
    def fit_transform(self, texts, y=None):
        """Fits and transforms the texts into sparse TF-IDF vectors in one step."""
        return self.vectorizer.fit_transform(texts)

    def get_feature_names(self):
        """Returns the vocabulary feature names."""
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            return self.vectorizer.get_feature_names_out()
        return self.vectorizer.get_feature_names()
