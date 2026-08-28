"""Aggregate experiment summaries into Excel/CSV from a manifest JSON.

Examples:
    uv run python summary.py results/experiments/curriculum_ablations_multi_20260828-014706.json

    # Legacy (deprecated): explicit folders or --run-prefix discovery
    uv run python summary.py --compare --run-prefix 20260711-022935 --datasets webkb reuters90
"""

from bio_is_curriculum.results.summary_export import main

if __name__ == "__main__":
    main()
