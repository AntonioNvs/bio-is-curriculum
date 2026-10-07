"""Backward-compatible entry point. Prefer: uv run bio-run ..."""

from bio_is_curriculum.cli.main import main

if __name__ == "__main__":
    main()
