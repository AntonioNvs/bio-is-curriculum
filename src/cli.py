import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import DatasetLoader
from data.preprocessor import TextPreprocessor

def main():
    parser = argparse.ArgumentParser(description="Load and preprocess text datasets.")
    parser.add_argument("dataset", type=str, help="Name of the dataset (e.g., ohsumed, mpqa, sst1)")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Directory where datasets are stored")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of splits/folds to load")
    parser.add_argument("--max_features", type=int, default=5000, help="Max features for TF-IDF")
    
    args = parser.parse_args()
    
    print(f"==========================================")
    print(f"Loading dataset: {args.dataset} from {args.data_dir}...")
    try:
        loader = DatasetLoader(data_dir=args.data_dir, dataset_name=args.dataset)
        texts, scores = loader.load_texts_and_scores()
        print(f"Successfully loaded {len(texts)} texts and {len(scores)} scores.")
        
        splits = loader.load_splits(n_splits=args.n_splits)
        print(f"Loaded {len(splits)} folds from split_{args.n_splits}.pkl")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"\nPreprocessing texts...");
    print(f"Applying TF-IDF (max_features={args.max_features})...")
    preprocessor = TextPreprocessor(max_features=args.max_features, lowercase=True)
    tfidf_matrix = preprocessor.fit_transform(texts)
    
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
    
    # Printing the first 10 vocabulary terms for demonstration
    sample_vocab = preprocessor.get_feature_names()[:10]
    print(f"Sample vocabulary extracted: {sample_vocab}")
    print(f"==========================================")

if __name__ == "__main__":
    main()
