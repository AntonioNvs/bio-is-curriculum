import numpy as np
import pytest
from sklearn.datasets import make_classification

from bio_is_curriculum.selection.biois import BIOIS


@pytest.fixture
def sparse_data():
    X, y = make_classification(
        n_samples=200, n_features=50, n_informative=20,
        n_redundant=10, n_classes=3, random_state=0,
    )
    return X, y


def test_biois_fit_reduces(sparse_data):
    X, y = sparse_data
    sel = BIOIS(beta=0.3, theta=0.2, random_state=0)
    sel.fit(X, y)
    assert sel.reduction_ >= 0.0
    assert len(sel.sample_indices_) <= len(y)
    assert len(sel.sample_indices_) > 0
