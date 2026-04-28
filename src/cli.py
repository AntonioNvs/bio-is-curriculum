import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curriculum.biois_curriculum import BIOISCurriculum
from data.loader import DatasetLoader
from iSel.biois import BIOIS


def main():
    parser = argparse.ArgumentParser(
        description="Load a prebuilt per-fold TF-IDF dataset and run BIOIS instance selection."
    )
    parser.add_argument("dataset", type=str, help="Name of the dataset (e.g., webkb, ohsumed, mpqa)")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Directory where datasets are stored")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to use as training set for BIOIS")
    parser.add_argument("--beta", type=float, default=0.0, help="BIOIS redundancy reduction rate (0 keeps all)")
    parser.add_argument("--theta", type=float, default=0.0, help="BIOIS noise reduction rate (0 keeps all)")
    parser.add_argument("--random-state", dest="random_state", type=int, default=42, help="Random seed for BIOIS")
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Run BIOIS-Curriculum staged training after BIOIS",
    )
    parser.add_argument(
        "--curriculum-beta",
        dest="curriculum_beta",
        type=float,
        default=0.5,
        help="Beta weight applied to redundant samples in the Hard phase (w_i = 1 - beta * r_i)",
    )
    parser.add_argument(
        "--curriculum-q",
        dest="curriculum_q",
        type=float,
        nargs=3,
        default=(0.3, 0.6, 0.95),
        metavar=("Q_LOW", "Q_MID", "Q_HIGH"),
        help="Entropy quantile thresholds for phases A, B, C (defaults: 0.3 0.6 0.95)",
    )

    args = parser.parse_args()

    print("==========================================")
    print(f"Loading prebuilt TF-IDF for dataset: {args.dataset} (fold {args.fold}) from {args.data_dir}...")
    try:
        loader = DatasetLoader(data_dir=args.data_dir, dataset_name=args.dataset)
        X_train, y_train, X_test, y_test = loader.load_tfidf_fold(args.fold)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print("Original dataset shape %s" % Counter(y_train.tolist()))
    print("==========================================")

    print(
        f"Running BIOIS (beta={args.beta}, theta={args.theta}, "
        f"random_state={args.random_state})..."
    )
    selector = BIOIS(beta=args.beta, theta=args.theta, random_state=args.random_state)
    selector.fit(X_train, y_train)

    idx = selector.sample_indices_
    X_train_selected, y_train_selected = X_train[idx], y_train[idx]
    print("Resampled dataset shape %s" % Counter(y_train_selected.tolist()))
    print(f"Reduction: {selector.reduction_:.4f}")
    print("==========================================")

    if args.curriculum:
        q_low, q_mid, q_high = args.curriculum_q
        print(
            f"Running BIOIS-Curriculum (beta={args.curriculum_beta}, "
            f"q_low={q_low}, q_mid={q_mid}, q_high={q_high})..."
        )
        curriculum = BIOISCurriculum(
            beta=args.curriculum_beta,
            q_low=q_low,
            q_mid=q_mid,
            q_high=q_high,
            random_state=args.random_state,
        )
        curriculum.fit(selector, X_train, y_train, X_test=X_test, y_test=y_test)

        print("Curriculum history (per phase):")
        for row in curriculum.history_:
            print(f"  {row}")
        print("==========================================")


if __name__ == "__main__":
    main()
