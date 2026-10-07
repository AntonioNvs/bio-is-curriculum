"""CUDA device selection — must run before PyTorch is imported."""

from __future__ import annotations

import os
import sys

from bio_is_curriculum.config.defaults import DEFAULTS


def running_in_docker() -> bool:
    """True when inside a container (GPU is chosen at ``docker run --gpus`` time)."""
    return os.path.exists("/.dockerenv") or os.environ.get("BIO_IS_IN_DOCKER") == "1"


def _parse_cuda_device_from_argv(argv: list[str]) -> int | None:
    for i, arg in enumerate(argv):
        if arg == "--cuda-device-id" and i + 1 < len(argv):
            return int(argv[i + 1])
        if arg.startswith("--cuda-device-id="):
            return int(arg.split("=", 1)[1])
    return None


def configure_cuda_device(device_id: int | None = None) -> int | None:
    """Pin training to a physical GPU via ``CUDA_VISIBLE_DEVICES``.

    Inside Docker, GPU selection is done on the **host** with
    ``docker run --gpus device=N``; this function does nothing there — the
    container sees a single GPU as ``cuda:0``.

    On bare-metal runs, defaults to physical GPU 7 unless ``CUDA_VISIBLE_DEVICES``
    is already set or ``--cuda-device-id`` is passed.

    Returns the device id applied, or ``None`` if pinning was skipped.
    """
    if running_in_docker():
        return None

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return None

    if device_id is None:
        device_id = _parse_cuda_device_from_argv(sys.argv)
    if device_id is None:
        device_id = DEFAULTS["cuda_device_id"]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    return device_id
