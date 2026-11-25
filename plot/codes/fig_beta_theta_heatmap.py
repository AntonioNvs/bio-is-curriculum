#!/usr/bin/env python
"""S5 figure: the Y5 beta x theta grid as a pair of 3x3 heatmaps.

Reads only the validated S4 output ``analysis/paper/out/numbers.json`` (the
same numbers the S4 table renders), so the heatmap and the LaTeX table cannot
drift apart.

Panels: Books and Reuters-90.
  3x3 cells: rows beta in {0.1, 0.3, 0.5}, cols theta in {0.0, 0.2, 0.4}.
  Colour  = delta macro-F1 vs the *default* cell (beta=0.3, theta=0.2) for the
            same dataset, on a diverging PuOr scale centred at 0.
  Label   = "F1 / retention" in each cell (e.g. "0.824 / 0.66").
  The default cell is outlined in black.

Y5 is IS-only, three folds, Books and Reuters-90 (plan V5/table tab:beta-theta).

Nothing here trains, fits, evaluates or selects.  ``torch`` / ``transformers``
are asserted absent by ``_style``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _style  # noqa: E402  (installs the no-torch guard)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

NAME = "fig_beta_theta_heatmap"
DATASETS = ["books", "reuters90"]
BETAS = ["0.1", "0.3", "0.5"]
THETAS = ["0.0", "0.2", "0.4"]
DEFAULT = ("0.3", "0.2")


def load_grid() -> dict:
    rows = json.loads((_style.OUT_DIR / "numbers.json").read_text())["rows"]
    grid: dict[str, dict[tuple[str, str], dict]] = {}
    for r in rows:
        if not r["key"].startswith("Y5/"):
            continue
        c = r["cells"]
        ds = c["dataset"]
        if ds not in DATASETS:
            continue
        beta = c["beta"]
        grid.setdefault(ds, {})
        for th in THETAS:
            cell = c[f"theta{th}"]
            grid[ds][(beta, th)] = {"f1": cell["f1"], "ret": cell["retention"]}
    for ds in DATASETS:
        assert ds in grid, f"missing Y5 grid for {ds}"
        assert len(grid[ds]) == 9, f"{ds}: expected 9 cells, got {len(grid[ds])}"
        assert DEFAULT in grid[ds], f"{ds}: missing default cell {DEFAULT}"
    return grid


def draw(grid: dict) -> plt.Figure:
    _style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), layout="constrained")

    for ax, ds in zip(axes, DATASETS):
        f1 = np.array([[grid[ds][(b, t)]["f1"] for t in THETAS] for b in BETAS])
        ret = np.array([[grid[ds][(b, t)]["ret"] for t in THETAS] for b in BETAS])
        ref = grid[ds][DEFAULT]["f1"]
        delta = f1 - ref
        vmax = max(abs(delta).max(), 1e-6)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        im = ax.imshow(delta, cmap="PuOr", norm=norm, origin="upper",
                       aspect="equal")
        for i, b in enumerate(BETAS):
            for j, t in enumerate(THETAS):
                ax.text(j, i, f"{f1[i, j]:.3f}\n/ {ret[i, j]:.2f}",
                        ha="center", va="center", fontsize=7.5,
                        color="black")
        # outline the default cell
        di = BETAS.index(DEFAULT[0])
        dj = THETAS.index(DEFAULT[1])
        ax.add_patch(Rectangle((dj - 0.5, di - 0.5), 1, 1, fill=False,
                               edgecolor="black", lw=1.6, zorder=3))
        ax.set_xticks(range(3), [f"$\\theta{{=}}{t}$" for t in THETAS])
        ax.set_yticks(range(3), [f"$\\beta{{=}}{b}$" for b in BETAS])
        ax.set_xlabel("Noise-risk $\\theta$")
        if ax is axes[0]:
            ax.set_ylabel("Selection rate $\\beta$")
        ax.set_title(_style.DATASET_LABEL[ds])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("$\\Delta$ macro-F1 vs default", fontsize=7.5)

    return fig


def main() -> None:
    grid = load_grid()
    _style.assert_no_ml()
    fig = draw(grid)
    _style.save_fig(fig, NAME)
    plt.close(fig)
    _style.assert_no_ml()
    print(f"{NAME}: {len(DATASETS)} panels x 9 cells, default outlined")


if __name__ == "__main__":
    main()
