import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: integration tests (may need datasets/GPU)")
    config.addinivalue_line("markers", "slow: slow tests (RoBERTa training)")
