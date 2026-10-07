"""Backward-compatible YAML runner. Prefer: uv run bio-experiment ..."""

from bio_is_curriculum.cli.experiment import main

if __name__ == "__main__":
    main()
