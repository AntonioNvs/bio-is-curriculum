"""Map each paper-table cell to its most recent result directory.

Usage:
    uv run python scripts/identify_latest.py [--output mapping.csv]
"""

from bio_is_curriculum.results.discovery import main

if __name__ == "__main__":
    main()
