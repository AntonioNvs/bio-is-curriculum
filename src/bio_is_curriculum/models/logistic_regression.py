"""Logistic regression backend for smoke tests and fast iteration."""

import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression

from bio_is_curriculum.models.base import CurriculumModel


def sklearn_at_least(major: int, minor: int) -> bool:
    toks = sk.__version__.split(".")[:4]

    def _digits(s: str) -> int:
        d = "".join(ch for ch in s if ch.isdigit())
        return int(d) if d else 0

    mj = _digits(toks[0])
    mn = _digits(toks[1]) if len(toks) > 1 else 0
    return (mj, mn) >= (major, minor)


def logistic_regression_user_spec(**kwargs) -> LogisticRegression:
    params: dict = {"C": 1.0, "solver": "saga", "max_iter": 1000}
    params.update(kwargs)
    return LogisticRegression(**params)


class LogisticRegressionModel(CurriculumModel):
    """Sklearn LR with warm_start for phased curriculum training."""

    def __init__(self, max_iter: int = 100, random_state: int = 42, **kwargs):
        self.max_iter = max_iter
        self.random_state = random_state
        self._clf = logistic_regression_user_spec(
            warm_start=True,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs,
        )

    def fit_stage(self, X, y, sample_weight=None, X_val=None, y_val=None, balanced_sampling=False):
        self._clf.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    @property
    def n_iter(self) -> int:
        return int(np.max(self._clf.n_iter_))
