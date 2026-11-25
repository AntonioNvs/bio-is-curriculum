#!/usr/bin/env python
"""Shared plotting style for the S5 figures (colourblind-safe palette + shapes).

Read-only post-processing helper.  Every figure script imports this module
first, so the no-torch / no-transformers guard here covers the whole S5 family:

    * at import it asserts neither ``torch`` nor ``transformers`` is already in
      ``sys.modules`` (they would mean an experiment was loaded in this
      process);
    * :func:`assert_no_ml` re-checks the same invariant before every save.

The palette is Okabe-Ito with two extra Tol entries, so the seven end points of
:mod:`fig_pareto_cost_f1` stay distinct for deuteranopes/protanopes and for
greyscale print.  Colour is never the only encoding: each series also carries a
marker shape (see :data:`SHAPE`).

Nothing here reads data or writes figures by itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- hard no-torch / no-transformers guard --------------------------------
for _mod in ("torch", "transformers"):
    assert _mod not in sys.modules, f"{_mod} must not be imported by an S5 script"

import matplotlib

matplotlib.use("Agg")  # headless, vector+PNG only
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# plot/codes/_style.py -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "analysis" / "paper" / "out"
IMAGES_DIR = REPO_ROOT / "plot" / "images"

# --------------------------------------------------------------------------- #
# Palette (Okabe-Ito + Tol) and marker shapes
# --------------------------------------------------------------------------- #

# Okabe-Ito core, extended with two Tol greys/browns for the 7-endpoint figure.
OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}
TOL_EXTRA = {"brown": "#8C510A", "grey": "#999999"}

# Fixed series colours for the Pareto figure.
SERIES_COLOR = {
    "Raw@2": "#FBE6B5",   # same hue family as Raw@6, light -> dark
    "Raw@4": "#F2C14E",
    "Raw@6": "#B8860B",
    "IS": "#0072B2",      # blue
    "CL": "#D55E00",      # vermillion
    "SPURS": "#000000",   # black star (the proposal)
    "Random@ret": "#CC79A7",  # reddish purple
}

# Marker shapes: colour + shape so the figure survives greyscale.
SHAPE = {
    "Raw@2": "o",
    "Raw@4": "o",
    "Raw@6": "o",
    "IS": "s",
    "CL": "^",
    "SPURS": "*",
    "Random@ret": "D",
}
MARKERSIZE = {"SPURS": 14.0, "Raw@2": 7.0, "Raw@4": 7.5, "Raw@6": 8.0}
DEFAULT_MARKERSIZE = 7.5

# Dataset colours for the calibration figure.
DATASET_COLOR = {
    "reuters90": "#0072B2",
    "agnews": "#D55E00",
    "yelp_2013": "#009E73",
    "books": "#CC79A7",
    "dblp": "#56B4E9",
    "medline": "#999999",
}
DATASET_MARKER = {"cl": "^", "is_cl": "*"}
ARM_LABEL = {"cl": "CL", "is_cl": "SPURS"}

DATASET_LABEL = {
    "yelp_2013": "Yelp-2013",
    "agnews": "AG News",
    "dblp": "DBLP",
    "books": "Books",
    "reuters90": "Reuters-90",
    "medline": "Medline",
}

# --------------------------------------------------------------------------- #
# rcParams: sans 8-9 pt, no chart junk
# --------------------------------------------------------------------------- #


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": False,
        "mathtext.fontset": "dejavusans",
        # Embed text as Type 42 (TrueType) rather than Type 3 in the vector
        # outputs, so the PDFs are portable and the glyphs are not bitmaps.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def assert_no_ml() -> None:
    """Re-check no torch/transformers is loaded, before any figure is saved."""
    for _mod in ("torch", "transformers"):
        assert _mod not in sys.modules, f"{_mod} must not be imported by an S5 script"


# --------------------------------------------------------------------------- #
# Save helper: 300-dpi PNG + vector PDF under plot/images/<name>/
# --------------------------------------------------------------------------- #


def save_fig(fig, name: str, variant: str | None = None) -> list[Path]:
    """Save ``fig`` as ``plot/images/<name>/<name>[__variant].{png,pdf}``.

    Returns the list of written paths.  Asserts no ML module slipped in.
    """
    assert_no_ml()
    stem = name if variant is None else f"{name}__{variant}"
    out_dir = IMAGES_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, format="png", dpi=300)
    fig.savefig(pdf, format="pdf")
    written = [png, pdf]
    for p in written:
        assert p.exists() and p.stat().st_size > 0, f"empty figure file: {p}"
    print(f"wrote {png.relative_to(REPO_ROOT)}  ({png.stat().st_size} B)")
    print(f"wrote {pdf.relative_to(REPO_ROOT)}  ({pdf.stat().st_size} B)")
    return written


__all__ = [
    "REPO_ROOT", "OUT_DIR", "IMAGES_DIR",
    "OKABE_ITO", "TOL_EXTRA", "SERIES_COLOR", "SHAPE", "MARKERSIZE",
    "DEFAULT_MARKERSIZE", "DATASET_COLOR", "DATASET_MARKER", "ARM_LABEL",
    "DATASET_LABEL",
    "apply_style", "assert_no_ml", "save_fig",
]
