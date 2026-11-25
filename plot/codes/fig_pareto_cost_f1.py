#!/usr/bin/env python
"""S5 figure: macro-F1 against end-to-end cost (Pareto scatter, no lines).

Reads only the validated S1 index ``analysis/paper/out/runs_index.csv``.

Panels: AG News, Books, Reuters-90 (one per dataset).
  x = fold-mean ``total_run_time_s``  (seconds, log scale)
  y = fold-mean ``f1_final`` (macro-F1) with a 95% t half-width error bar
  7 end points, all separate runs:
    Raw@2 / Raw@4 / Raw@6 (Y2 / Y2 / Y1 raw), one hue light -> dark
    IS (Y1 is)  ■
    CL (Y1 cl)  ▲
    SPURS (Y1 is_cl)  ★
    Random@ret (Y4 raw_rand<ret>)  ◆
No lines are drawn: the points are end points of separate runs, not a curve.
A dashed vertical guide marks the Raw@6 (full-budget) cost per panel; it is a
reference line, not a series.

Variants:
  fig_pareto_cost_f1__adjusted : x = total_run_time_s - nonfinal_pred_time_s
      (removes the in-loop non-final evaluations CL/SPURS pay, V9).

Nothing here trains, fits, evaluates or selects.  ``torch`` / ``transformers``
are asserted absent by ``_style``.
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

NAME = "fig_pareto_cost_f1"
PANELS = ["agnews", "books", "reuters90"]
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}

# (label, campaign, arm-regex, colour-key, shape-key)
ENDPOINTS = [
    ("Raw@2", "Y2", r"^raw_ep2$"),
    ("Raw@4", "Y2", r"^raw_ep4$"),
    ("Raw@6", "Y1", r"^raw$"),
    ("IS", "Y1", r"^is$"),
    ("CL", "Y1", r"^cl$"),
    ("SPURS", "Y1", r"^is_cl$"),
    ("Random@ret", "Y4", r"^raw_rand"),
]


def t_halfwidth(vals: np.ndarray) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    se = float(np.std(vals, ddof=1)) / math.sqrt(n)
    return T_CRIT_95.get(n - 1, 1.96) * se


def nice_ticks(lo: float, hi: float, k: int = 4) -> list[float]:
    """Evenly log-spaced ticks in [lo, hi], rounded to 2 significant figures.

    Round decade multiples bunch at the top of a wide range, so space the
    ticks geometrically first and round them afterwards.
    """
    ticks: list[float] = []
    for v in np.geomspace(lo, hi, k):
        e = math.floor(math.log10(v))
        ticks.append(round(float(v), -(e - 1)))
    return sorted(set(ticks))


def load_points() -> pd.DataFrame:
    idx = pd.read_csv(_style.OUT_DIR / "runs_index.csv")
    rows = []
    for label, campaign, arm_re in ENDPOINTS:
        for ds in PANELS:
            sub = idx[(idx.campaign == campaign) & (idx.dataset == ds)
                      & idx.arm.str.match(arm_re)]
            if sub.empty:
                raise SystemExit(f"MISSING {label} for {ds} ({campaign} ~ {arm_re})")
            f1 = sub["f1_final"].to_numpy(dtype=float)
            rows.append({
                "dataset": ds,
                "series": label,
                "n_folds": len(sub),
                "x_total_s": float(sub["total_run_time_s"].mean()),
                "nonfinal_pred_s": float(sub["nonfinal_pred_time_s"].mean()),
                "y_f1": float(f1.mean()),
                "y_half": t_halfwidth(f1),
                "retention": float(sub["retention_effective"].mean()),
            })
    return pd.DataFrame(rows)


def draw(pts: pd.DataFrame, adjusted: bool) -> plt.Figure:
    _style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))

    for ax, ds in zip(axes, PANELS):
        sub = pts[pts.dataset == ds]
        x = (sub["x_total_s"] - sub["nonfinal_pred_s"]) if adjusted else sub["x_total_s"]
        for _, r in sub.iterrows():
            lab = r["series"]
            xv = (r["x_total_s"] - r["nonfinal_pred_s"]) if adjusted else r["x_total_s"]
            # draw a zero-half-width bar as a tiny cap, kept for uniformity
            ax.errorbar(
                xv, r["y_f1"], yerr=r["y_half"],
                fmt=_style.SHAPE[lab], color=_style.SERIES_COLOR[lab],
                markersize=_style.MARKERSIZE.get(lab, _style.DEFAULT_MARKERSIZE),
                markeredgecolor="black", markeredgewidth=0.4,
                elinewidth=0.8, capsize=1.8, capthick=0.6, linestyle="none",
                zorder=3,
            )
        # Reference: the full-budget Raw@6 cost (dashed vertical guide).
        raw6 = sub.loc[sub.series == "Raw@6"].iloc[0]
        x_ref = (raw6["x_total_s"] - raw6["nonfinal_pred_s"]) if adjusted else raw6["x_total_s"]
        ax.axvline(x_ref, color="#999999", lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.set_xscale("log")
        ax.set_title(_style.DATASET_LABEL[ds])
        ax.set_xlabel("Time per fold (s, log)")
        # Explicit round-second ticks; auto-scaled log decade labels collide
        # on these narrow ranges, so choose a handful inside the data envelope
        # and format them as plain integers.
        lo, hi = float(x.min()), float(x.max())
        ticks = nice_ticks(lo, hi, k=3)
        ax.set_xlim(min(lo, ticks[0]) * 0.90, max(hi, ticks[-1]) * 1.10)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:,.0f}" for t in ticks])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(True, which="major", axis="both")
        ax.margins(y=0.18)
    axes[0].set_ylabel("Macro-F1 (fold mean $\\pm$95\\% CI)")

    handles = [
        mlines.Line2D([], [], marker=_style.SHAPE[l], linestyle="none",
                      color=_style.SERIES_COLOR[l],
                      markersize=_style.MARKERSIZE.get(l, _style.DEFAULT_MARKERSIZE),
                      markeredgecolor="black", markeredgewidth=0.4, label=l)
        for l, _, _ in ENDPOINTS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def main() -> None:
    pts = load_points()
    _style.assert_no_ml()
    fig = draw(pts, adjusted=False)
    _style.save_fig(fig, NAME)
    plt.close(fig)

    fig = draw(pts, adjusted=True)
    _style.save_fig(fig, NAME, variant="adjusted")
    plt.close(fig)
    _style.assert_no_ml()
    print(f"{NAME}: {len(pts)} end points over {len(PANELS)} panels")


if __name__ == "__main__":
    main()
