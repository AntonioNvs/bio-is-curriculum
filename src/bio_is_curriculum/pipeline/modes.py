"""Mode constants and dispatch helpers."""

import re

IS_MODES = frozenset({"is", "is_cl", "is_continuous_cl", "is_continuos_cl"})
CL_MODES = frozenset({"cl", "is_cl", "is_continuous_cl", "is_continuos_cl"})
IS_CL_MODES = frozenset({"is_cl", "is_continuous_cl", "is_continuos_cl"})
BASELINE_MODE_RE = re.compile(r"^b([0-9]+)$")
IS_BASELINE_MODE_RE = re.compile(r"^is_b([0-9]+)$")

MODE_ALIASES = {
    "is_continuos_cl": "is_continuous_cl",
}


def normalize_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def parse_baseline_index(mode: str) -> int | None:
    m = BASELINE_MODE_RE.match(mode)
    return int(m.group(1)) if m else None


def parse_is_baseline_index(mode: str) -> int | None:
    """Baseline on BIOIS-selected data only (IS is data prep, not the baseline signal)."""
    m = IS_BASELINE_MODE_RE.match(mode)
    return int(m.group(1)) if m else None


def uses_is_subset(mode: str) -> bool:
    return mode in IS_CL_MODES or parse_is_baseline_index(mode) is not None
