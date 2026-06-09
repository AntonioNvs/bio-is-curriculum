"""Base para baselines de currículo da literatura.

Cada baseline herda de `BaselineBase` (que reaproveita o orquestrador
faseado do `BIOISDiscreteCurriculum` — modelo + fases + recorder + métricas),
e sobrescreve apenas a *extração de sinais* e a *construção das fases*
conforme a definição original do método.

Convenção:
- `INDEX` (int): identificador estável usado em `--baseline N` e em
  `BASELINES.md`. Não reutilize índices entre baselines.
- `NAME` (str): nome curto para logs/títulos.
- `REFERENCE` (str): citação completa (autor, ano, venue, DOI/URL).
"""
from abc import ABCMeta

from curriculum.methods.biois_discrete import BIOISDiscreteCurriculum


class BaselineBase(BIOISDiscreteCurriculum, metaclass=ABCMeta):
    """Interface comum aos baselines de CL da literatura."""

    INDEX: int = -1
    NAME: str = "abstract"
    REFERENCE: str = ""
