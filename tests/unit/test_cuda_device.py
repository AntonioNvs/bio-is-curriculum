"""Tests for CUDA device configuration."""

import os

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
