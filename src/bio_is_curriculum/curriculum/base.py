"""Base para metodos de curriculum learning."""
from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator


class CurriculumBase(BaseEstimator, metaclass=ABCMeta):
    """Mixin base para estrategias de curriculum learning.

    Recebe um seletor de instancias ja ajustado (que carrega os sinais
    estruturais do dataset) e organiza temporalmente o treinamento.
    """

    @abstractmethod
    def fit(self, selector, X, y, X_test=None, y_test=None):
        """Executa o curriculum a partir de um seletor ja ajustado."""
        ...
