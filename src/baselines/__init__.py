"""Registry de baselines de CL da literatura.

Cada baseline e indexado por um inteiro estavel (`--baseline N`) e
documentado em `BASELINES.md` na raiz do projeto.
"""
from .base import BaselineBase
from .baseline1 import Baseline1

REGISTRY: dict[int, type[BaselineBase]] = {
    Baseline1.INDEX: Baseline1,
}


def get_baseline(index: int) -> type[BaselineBase]:
    """Retorna a classe do baseline correspondente ao indice, ou levanta."""
    if index not in REGISTRY:
        disponiveis = sorted(REGISTRY)
        raise ValueError(
            f"Baseline {index} nao encontrado. Disponiveis: {disponiveis}. "
            "Consulte BASELINES.md para a descricao de cada indice."
        )
    return REGISTRY[index]


def baseline_run_id(index: int) -> str:
    """Slug usado em nomes de pasta de resultados: ex. b1, b2, ..."""
    return f"b{index}"


__all__ = ["BaselineBase", "Baseline1", "REGISTRY", "get_baseline", "baseline_run_id"]
