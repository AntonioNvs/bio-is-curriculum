"""Abstracoes de modelo usadas pelo curriculum learning.

A logica do curriculum (`BIOISCurriculum`) interage com um modelo
unicamente atraves desta interface, de modo que novos backends
(p. ex. transformers, SGD, etc.) possam ser plugados sem alterar a
construcao das fases ou o registro de metricas.
"""
from abc import ABCMeta, abstractmethod

import inspect

import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression


def sklearn_at_least(major: int, minor: int) -> bool:
    toks = sk.__version__.split(".")[:4]
    def _digits(s: str) -> int:
        d = "".join(ch for ch in s if ch.isdigit())
        return int(d) if d else 0

    mj = _digits(toks[0])
    mn = _digits(toks[1]) if len(toks) > 1 else 0
    return (mj, mn) >= (major, minor)


def logistic_regression_user_spec(**kwargs) -> LogisticRegression:
    """Defaults robustos para texto esparso e cenarios multiclasses."""
    params: dict = {
        "C": 1.0,
        "solver": "saga",
        "max_iter": 1000
    }
    params.update(kwargs)
    return LogisticRegression(**params)


class CurriculumModel(metaclass=ABCMeta):
    """Interface minima exigida pelo curriculum.

    Implementacoes devem manter estado entre chamadas de `fit_stage`
    para que o treinamento seja efetivamente faseado (continuado), e
    nao um re-treino do zero a cada fase.
    """

    @abstractmethod
    def fit_stage(self, X, y, sample_weight=None):
        """Continua o treinamento por mais uma fase do curriculum."""
        ...

    @abstractmethod
    def predict(self, X):
        ...

    @abstractmethod
    def predict_proba(self, X):
        ...

    @property
    @abstractmethod
    def n_iter(self) -> int:
        """Numero de iteracoes executadas na ultima fase (proxy de steps)."""
        ...


class LogisticRegressionModel(CurriculumModel):
    """Implementacao default baseada em `LogisticRegression(warm_start=True)`.

    O `warm_start` preserva os coeficientes entre chamadas de
    `fit_stage`, garantindo que cada fase do curriculum continue o
    treinamento da fase anterior.
    """

    def __init__(self, max_iter: int = 100, random_state: int = 42, **kwargs):
        self.max_iter = max_iter
        self.random_state = random_state
        self._clf = logistic_regression_user_spec(
            warm_start=True,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs,
        )

    def fit_stage(self, X, y, sample_weight=None):
        self._clf.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    @property
    def n_iter(self) -> int:
        return int(np.max(self._clf.n_iter_))
