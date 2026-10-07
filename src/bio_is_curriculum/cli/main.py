"""Single-fold CLI entry point."""

from __future__ import annotations

import os

# Determinism and GPU selection before torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")

from bio_is_curriculum.config.cuda import configure_cuda_device

configure_cuda_device()

from bio_is_curriculum.cli.args import args_to_config, build_parser
from bio_is_curriculum.pipeline.runner import run_experiment


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
