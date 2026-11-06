"""Tests for CUDA device configuration."""

import os
import sys

import pytest

from bio_is_curriculum.config.cuda import configure_cuda_device


@pytest.fixture
def clean_cuda_env(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("BIO_IS_IN_DOCKER", raising=False)


def test_configure_cuda_device_default(clean_cuda_env):
    applied = configure_cuda_device()
    assert applied == 7
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"


def test_configure_cuda_device_skips_in_docker(clean_cuda_env, monkeypatch):
    monkeypatch.setenv("BIO_IS_IN_DOCKER", "1")
    applied = configure_cuda_device()
    assert applied is None
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_configure_cuda_device_respects_existing_env(clean_cuda_env, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    applied = configure_cuda_device(7)
    assert applied is None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_configure_cuda_device_explicit_override(clean_cuda_env):
    applied = configure_cuda_device(3)
    assert applied == 3
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


# --------------------------------------------------------------------------- #
# The RunPod path: cli/experiment.py passes --cuda-device-id to every fold
# subprocess, so the pin is triggered through argv, not through an argument.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.path.exists("/.dockerenv"),
    reason="inside a container, so running_in_docker() short-circuits the pin",
)
def test_argv_cuda_device_id_is_honoured(clean_cuda_env, monkeypatch):
    """Proves the live trigger: argv alone is enough to pin a physical GPU."""
    monkeypatch.setattr(sys, "argv", ["main", "--cuda-device-id", "7"])
    applied = configure_cuda_device()
    assert applied == 7
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"


@pytest.mark.skipif(
    os.path.exists("/.dockerenv"),
    reason="inside a container, so running_in_docker() short-circuits the pin",
)
def test_bootstrap_env_block_defuses_the_argv_pin(clean_cuda_env, monkeypatch):
    """The RunPod guard: with the pod env block, argv cannot repin the device.

    Without this, a 1-GPU pod would hide its only GPU behind physical device 7
    and train on CPU, with no error — indistinguishable from a slow run.
    """
    monkeypatch.setenv("BIO_IS_IN_DOCKER", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(sys, "argv", ["main", "--cuda-device-id", "7"])

    applied = configure_cuda_device()
    assert applied is None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.skipif(
    os.path.exists("/.dockerenv"),
    reason="inside a container, so running_in_docker() short-circuits the pin",
)
def test_argv_flag_with_equals_form(clean_cuda_env, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main", "--cuda-device-id=5"])
    assert configure_cuda_device() == 5
