"""Specification of the paper's result tables.

Each entry maps a cell in one of the LaTeX tables of ``acl_latex (1).tex`` to
the ``(mode, curriculum_method, extra config fields)`` that identify the run
producing it. The ``match`` dict disambiguates runs that share the same
``mode``/``curriculum_method`` but differ in scheduler configuration (e.g. the
CL parameter ablation, whose variants all use ``cl`` + ``biois_discrete``).

These specs are the single source of truth consumed by both
``discovery.identify_latest`` (latest-run resolution) and
``significance.run`` (paired statistical tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field

FACTORIAL_DATASETS = [
    "reuters90",
    "dblp",
    "books",
    "yelp_2013",
    "agnews",
    "medline",
]
# The signal/params ablations never include medline.
ABLATION_DATASETS = [
    "reuters90",
    "dblp",
    "books",
    "yelp_2013",
    "agnews",
]

# Config fields that every run feeding a paper table must agree on. The 2^2
# factorial is only interpretable if its four cells share the encoder and the
# batch size; the repository contains older runs that do not (``roberta-base``
# at ``batch_size`` 16 next to ``ModernBERT-base`` at 32), and resolving those
# silently is what produced the unmatched cells this list now rejects.
REQUIRED_CONFIG = {
    "hf_model": "answerdotai/ModernBERT-base",
    "batch_size": 32,
}

# Canonical BIO-IS discrete curriculum (the paper's "Default" configuration).
_DEFAULT_MATCH = {
    "curriculum_q": [0.3, 0.6, 0.95],
    "curriculum_margin_weight": 0.6,
    "curriculum_entropy_weight": 0.4,
    "curriculum_phase_max_lengths": [96, 160, 256],
}


@dataclass(frozen=True)
class MethodSpec:
    """One row/method of a paper table.

    ``match`` holds config fields (compared by value, list-aware) that further
    disambiguate runs sharing ``mode`` and ``curriculum_method``.
    """

    name: str
    mode: str
    curriculum_method: str
    match: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TableSpec:
    """A paper table: a set of methods evaluated on a set of datasets."""

    name: str
    datasets: list[str]
    methods: list[MethodSpec]


def _m(
    name: str,
    mode: str,
    curriculum_method: str = "biois_discrete",
    match: dict | None = None,
) -> MethodSpec:
    return MethodSpec(name, mode, curriculum_method, dict(match or {}))


PAPER_TABLES: list[TableSpec] = [
    TableSpec(
        name="factorial",
        datasets=FACTORIAL_DATASETS,
        methods=[
            _m("Raw", "raw"),
            _m("IS", "is"),
            _m("CL", "cl", match=_DEFAULT_MATCH),
            _m("IS+CL", "is_cl", match=_DEFAULT_MATCH),
        ],
    ),
    TableSpec(
        name="signal_ablation",
        datasets=ABLATION_DATASETS,
        methods=[
            _m("BIO-IS", "cl", match=_DEFAULT_MATCH),
            _m("Loss", "cl", curriculum_method="loss_discrete"),
            _m("LRC", "cl", curriculum_method="lrc_discrete"),
            _m("TD", "cl", curriculum_method="td_discrete"),
        ],
    ),
    TableSpec(
        name="params_ablation",
        datasets=ABLATION_DATASETS,
        methods=[
            _m("Default", "cl", match=_DEFAULT_MATCH),
            _m(
                "Tight schedule",
                "cl",
                match={
                    "curriculum_q": [0.2, 0.5, 0.95],
                    "curriculum_margin_weight": 0.6,
                    "curriculum_entropy_weight": 0.4,
                    "curriculum_phase_max_lengths": [96, 160, 256],
                },
            ),
            _m(
                "Entropy-heavy",
                "cl",
                match={
                    "curriculum_q": [0.3, 0.6, 0.95],
                    "curriculum_margin_weight": 0.3,
                    "curriculum_entropy_weight": 0.7,
                    "curriculum_phase_max_lengths": [96, 160, 256],
                },
            ),
            _m(
                "Flat compute",
                "cl",
                match={
                    "curriculum_q": [0.3, 0.6, 0.95],
                    "curriculum_margin_weight": 0.6,
                    "curriculum_entropy_weight": 0.4,
                    "curriculum_phase_max_lengths": [256, 256, 256],
                },
            ),
        ],
    ),
    TableSpec(
        name="baselines",
        datasets=FACTORIAL_DATASETS,
        methods=[
            _m("Bengio", "b1"),
            _m("BIOIS->Bengio", "is_b1"),
            _m("SPDCL", "b2"),
            _m("BIOIS->SPDCL", "is_b2"),
            _m("PMD", "cl", curriculum_method="self_adaptive_pmd"),
            _m("BIOIS->PMD", "is_cl", curriculum_method="self_adaptive_pmd"),
            _m("BIOIS-CL", "is_cl", match=_DEFAULT_MATCH),
        ],
    ),
]
