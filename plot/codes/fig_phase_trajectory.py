#!/usr/bin/env python
"""Appendix figure: per-phase curriculum trajectory against cumulative compute.

Reads only validated post-processing inputs:
  * ``analysis/paper/out/runs_index.csv`` (S1) for the run -> directory map;
  * each run's own ``phase_metrics.csv``, written by the training run.

Nothing here trains, fits, evaluates or selects; the phase files are only read.

For the Y1 matched factorial, every run logs one row per phase.  Raw and IS run
a single ``full`` phase; CL and SPURS run ``clean`` -> ``diverse`` -> ``hard``.
Because the three CL/SPURS rows are successive checkpoints of *one* run on a
cumulative compute axis, connecting them with a line is legitimate (unlike the
Pareto figure, whose points are separate runs).

Panels: the five matched datasets, one per cell, in ``dataset_order``.
  x = phase-mean cumulative ``compute_proxy`` (log scale)
  y = phase-mean macro-F1 with a 95% t half-width over the 5 folds
Series: CL (vermillion triangle) and SPURS (black star) as 3-point trajectories;
Raw (dark gold circle) and IS (blue square) as single end points.

Variants:
  fig_phase_trajectory__hard : y = ``hard_slice_macro_f1`` (rare-class slice).

``torch`` / ``transformers`` are asserted absent by ``_style``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _style  # noqa: E402  (installs the no-torch guard)

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NAME = "fig_phase_trajectory"
CAMPAIGN = "Y1"
PANELS = ["yelp_2013", "agnews", "dblp", "books", "reuters90"]
PHASE_RANK = {"clean": 0, "diverse": 1, "hard": 2, "full": 3}

# arm -> (label, colour, marker, is_trajectory)
ARMS = {
    "cl":    ("CL",    "#D55E00", "^", True),
    "is_cl": ("SPURS", "#000000", "*", True),
    "raw":   ("Raw",   "#B8860B", "o", False),
    "is":    ("IS",    "#0072B2", "s", False),
}
MARKERSIZE = {"is_cl": 11.0, "cl": 7.0, "raw": 7.5, "is": 7.0}
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def t_halfwidth(vals: np.ndarray) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    se = float(np.std(vals, ddof=1)) / math.sqrt(n)
    return T_CRIT_95.get(n - 1, 1.96) * se


def nice_ticks(lo: float, hi: float, k: int = 3) -> list[float]:
    """Evenly log-spaced ticks in [lo, hi], rounded to 2 significant figures.

    Matplotlib's default log labelling puts full mantissas on the minor ticks,
    which collide on these narrow per-panel ranges, so place a handful of ticks
    geometrically and round them.
    """
    ticks: list[float] = []
    for v in np.geomspace(lo, hi, k):
        e = math.floor(math.log10(v))
        ticks.append(round(float(v), -(e - 1)))
    return sorted(set(ticks))


def load_phases() -> pd.DataFrame:
    idx = pd.read_csv(_style.OUT_DIR / "runs_index.csv")
    sub = idx[(idx.campaign == CAMPAIGN) & idx.dataset.isin(PANELS)]
    rows = []
    for _, r in sub.iterrows():
        pm = _style.REPO_ROOT / str(r["dir"]) / "phase_metrics.csv"
        if not pm.exists():
            raise SystemExit(f"MISSING phase_metrics.csv for {r['dataset']}/{r['arm']}/fold{r['fold']}")
        ph = pd.read_csv(pm)
        for _, q in ph.iterrows():
            rows.append({
                "dataset": r["dataset"],
                "arm": r["arm"],
                "fold": int(r["fold"]),
                "phase": str(q["phase"]),
                "macro_f1": float(q["macro_f1"]),
                "hard_slice_macro_f1": float(q["hard_slice_macro_f1"]),
                "compute_proxy": float(q["compute_proxy"]),
            })
    long = pd.DataFrame(rows)
    agg = (
        long.groupby(["dataset", "arm", "phase"], as_index=False)
        .agg(x=("compute_proxy", "mean"),
             y_f1=("macro_f1", "mean"),
             y_hard=("hard_slice_macro_f1", "mean"),
             n_folds=("fold", "nunique"),
             f1_vals=("macro_f1", lambda s: list(s)),
             hard_vals=("hard_slice_macro_f1", lambda s: list(s)))
    )
    agg["half_f1"] = agg["f1_vals"].apply(lambda v: t_halfwidth(np.asarray(v, dtype=float)))
    agg["half_hard"] = agg["hard_vals"].apply(lambda v: t_halfwidth(np.asarray(v, dtype=float)))
    # Sanity: every trajectory arm must carry all three phases.
    for (ds, arm), g in agg.groupby(["dataset", "arm"]):
        want = {"clean", "diverse", "hard"} if ARMS[arm][3] else {"full"}
        if set(g["phase"]) != want:
            raise SystemExit(f"unexpected phases for {ds}/{arm}: {sorted(g['phase'])}")
    return agg


def draw(agg: pd.DataFrame, hard: bool) -> plt.Figure:
    _style.apply_style()
    ycol, ecol = ("y_hard", "half_hard") if hard else ("y_f1", "half_f1")
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3))
    axes = axes.ravel()

    for ax, ds in zip(axes, PANELS):
        panel = agg[agg.dataset == ds]
        for arm, (label, colour, marker, is_traj) in ARMS.items():
            g = panel[panel.arm == arm].copy()
            if g.empty:
                raise SystemExit(f"MISSING {arm} for {ds}")
            g["rank"] = g["phase"].map(PHASE_RANK)
            g = g.sort_values("rank")
            ax.errorbar(
                g["x"], g[ycol], yerr=g[ecol],
                fmt=marker, color=colour, linestyle="-" if is_traj else "none",
                markersize=MARKERSIZE.get(arm, 7.5),
                markeredgecolor="black", markeredgewidth=0.4,
                elinewidth=0.8, capsize=1.8, capthick=0.6,
                linewidth=1.0, zorder=3,
            )
        ax.set_xscale("log")
        ax.set_title(_style.DATASET_LABEL[ds], pad=3)
        ax.set_xlabel("Cumulative compute (log)")
        lo, hi = float(panel["x"].min()), float(panel["x"].max())
        ticks = nice_ticks(lo, hi, k=3)
        ax.set_xlim(min(lo, ticks[0]) * 0.92, max(hi, ticks[-1]) * 1.10)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:,.0f}" for t in ticks])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(True, which="major", axis="both")
        ax.margins(y=0.22)

    axes[0].set_ylabel("Macro-F1 (fold mean $\\pm$95\\% CI)")
    axes[3].set_ylabel("Macro-F1 (fold mean $\\pm$95\\% CI)")

    # Last cell: series legend (replaces the 6th panel).
    leg_ax = axes[5]
    leg_ax.axis("off")
    handles = [
        mlines.Line2D([], [], marker=ARMS[a][2], color=ARMS[a][1],
                      linestyle="-" if ARMS[a][3] else "none",
                      markersize=MARKERSIZE.get(a, 7.5),
                      markeredgecolor="black", markeredgewidth=0.4,
                      label=ARMS[a][0] + (" (3 phases)" if ARMS[a][3] else " (full)"))
        for a in ("raw", "is", "cl", "is_cl")
    ]
    leg_ax.legend(handles=handles, loc="center", frameon=False, handletextpad=0.6,
                  labelspacing=1.0)
    fig.tight_layout()
    return fig


def main() -> None:
    agg = load_phases()
    _style.assert_no_ml()
    for variant, hard in (("", False), ("hard", True)):
        fig = draw(agg, hard=hard)
        _style.save_fig(fig, NAME, variant=variant or None)
        plt.close(fig)
    _style.assert_no_ml()
    n_runs = int(agg["n_folds"].sum())
    print(f"{NAME}: {len(agg)} phase points over {len(PANELS)} panels "
          f"({n_runs} fold-phases)")


if __name__ == "__main__":
    main()
