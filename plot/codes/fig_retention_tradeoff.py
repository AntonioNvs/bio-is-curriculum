#!/usr/bin/env python
"""Appendix figure: macro-F1 against how much of the training set is kept.

Reads only ``analysis/paper/out/runs_index.csv`` (S1).  Nothing here trains,
fits, evaluates or selects.

Every figure in the paper reports accuracy at one retention.  This one puts the
whole removal range on a single axis, so the question "does removing more help?"
is answered by one curve instead of by three tables:

  * the Y5 beta x theta grid (9 cells x 3 folds) - IS only, no curriculum,
    covering retention ~0.90 down to ~0.40;
  * the Y4 random-subset control, run at exactly the retention the IS arm
    reached, so it separates "these instances" from "this fraction";
  * the Y1 matched arms IS (blue square) and SPURS (black star);
  * the Y1 Raw endpoint as a dashed horizontal reference at retention 1.0.

Panels: Books and Reuters-90, the two datasets the Y5 grid covers.
  x = ``retention_effective`` (fraction of instances kept)
  y = fold-mean macro-F1 (variant ``__hard``: rare-class slice)

Variants:
  fig_retention_tradeoff__hard : y = ``hard_f1``.

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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NAME = "fig_retention_tradeoff"
PANELS = ["books", "reuters90"]

BETA_COLOR = {0.1: "#E69F00", 0.3: "#0072B2", 0.5: "#009E73"}  # orange / blue / green
THETA_SHAPE = {0.0: "o", 0.2: "s", 0.4: "^"}

IS_COLOR, IS_SHAPE = "#000000", "s"
SPURS_COLOR, SPURS_SHAPE = "#000000", "*"
RANDOM_COLOR, RANDOM_SHAPE = "#CC79A7", "D"
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def t_halfwidth(vals: np.ndarray) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    se = float(np.std(vals, ddof=1)) / math.sqrt(n)
    return T_CRIT_95.get(n - 1, 1.96) * se


def pts(idx: pd.DataFrame, campaign: str, arm: str, ds: str, ycol: str) -> dict:
    sub = idx[(idx.campaign == campaign) & (idx.arm == arm) & (idx.dataset == ds)]
    if sub.empty:
        raise SystemExit(f"MISSING {campaign}/{arm} for {ds}")
    v = sub[ycol].to_numpy(dtype=float)
    return {
        "x": float(sub["retention_effective"].mean()),
        "y": float(v.mean()),
        "half": t_halfwidth(v),
        "n": len(sub),
    }


def load() -> dict:
    idx = pd.read_csv(_style.OUT_DIR / "runs_index.csv")
    out: dict = {"sweep": [], "arms": {}, "random": {}, "raw": {}}
    for ds in PANELS:
        sweep = idx[(idx.campaign == "Y5") & (idx.dataset == ds)]
        if sweep.empty:
            raise SystemExit(f"MISSING Y5 sweep for {ds}")
        for _, g in sweep.groupby("arm"):
            v = g["f1_final"].to_numpy(dtype=float)
            h = g["hard_f1"].to_numpy(dtype=float)
            out["sweep"].append({
                "dataset": ds,
                "beta": float(g["beta"].iloc[0]),
                "theta": float(g["theta"].iloc[0]),
                "x": float(g["retention_effective"].mean()),
                "y_f1": float(v.mean()), "half_f1": t_halfwidth(v),
                "y_hard": float(h.mean()), "half_hard": t_halfwidth(h),
                "n_folds": len(g),
            })
        out["arms"][ds] = {
            "IS": pts(idx, "Y1", "is", ds, "f1_final"),
            "SPURS": pts(idx, "Y1", "is_cl", ds, "f1_final"),
        }
        out["arms"][ds]["IS"]["hard"] = pts(idx, "Y1", "is", ds, "hard_f1")
        out["arms"][ds]["SPURS"]["hard"] = pts(idx, "Y1", "is_cl", ds, "hard_f1")
        rnd = idx[(idx.campaign == "Y4") & (idx.dataset == ds)]
        if rnd.empty:
            raise SystemExit(f"MISSING Y4 random control for {ds}")
        out["random"][ds] = {
            "x": float(rnd["retention_effective"].mean()),
            "y_f1": float(rnd["f1_final"].mean()),
            "half_f1": t_halfwidth(rnd["f1_final"].to_numpy(dtype=float)),
            "y_hard": float(rnd["hard_f1"].mean()),
            "half_hard": t_halfwidth(rnd["hard_f1"].to_numpy(dtype=float)),
            "n_folds": len(rnd),
        }
        out["raw"][ds] = {
            "y_f1": float(idx[(idx.campaign == "Y1") & (idx.arm == "raw")
                              & (idx.dataset == ds)]["f1_final"].mean()),
            "y_hard": float(idx[(idx.campaign == "Y1") & (idx.arm == "raw")
                                & (idx.dataset == ds)]["hard_f1"].mean()),
        }
    return out


def draw(data: dict, hard: bool) -> plt.Figure:
    _style.apply_style()
    yk, hk = ("y_hard", "half_hard") if hard else ("y_f1", "half_f1")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    for ax, ds in zip(axes, PANELS):
        # Raw reference: retention 1.0, dashed horizontal line.
        ax.axhline(data["raw"][ds][yk], color="#999999", lw=0.7, ls=(0, (4, 3)), zorder=1)

        sw = [s for s in data["sweep"] if s["dataset"] == ds]
        for s in sw:
            ax.errorbar(s["x"], s[yk], yerr=s[hk],
                        fmt=THETA_SHAPE[s["theta"]], color=BETA_COLOR[s["beta"]],
                        linestyle="none", markersize=6.0,
                        markeredgecolor="black", markeredgewidth=0.4,
                        elinewidth=0.7, capsize=1.5, capthick=0.5, zorder=2)

        r = data["random"][ds]
        ax.errorbar(r["x"], r[yk], yerr=r[hk], fmt=RANDOM_SHAPE, color=RANDOM_COLOR,
                    linestyle="none", markersize=6.5, markeredgecolor="black",
                    markeredgewidth=0.4, elinewidth=0.8, capsize=1.6, capthick=0.6, zorder=3)

        for name, colour, shape, size in (("IS", IS_COLOR, IS_SHAPE, 8.0),
                                          ("SPURS", SPURS_COLOR, SPURS_SHAPE, 13.0)):
            p = data["arms"][ds][name]
            y, half = (p["hard"]["y"], p["hard"]["half"]) if hard else (p["y"], p["half"])
            ax.errorbar(p["x"], y, yerr=half, fmt=shape, color=colour,
                        linestyle="none", markersize=size,
                        markeredgecolor="black", markeredgewidth=0.5,
                        elinewidth=0.9, capsize=1.8, capthick=0.6, zorder=4)

        ax.set_title(_style.DATASET_LABEL[ds], pad=3)
        ax.set_xlabel("Retention (fraction of instances kept)")
        ax.set_xlim(0.32, 1.04)
        ax.grid(True, which="major", axis="both")
        ax.margins(y=0.16)

    axes[0].set_ylabel("Macro-F1 (fold mean $\\pm$95\\% CI)")

    # Compact key: the 9-cell grid is read as colour (beta) x marker (theta),
    # so it needs 6 entries, not one per cell.
    beta_handles = [
        mlines.Line2D([], [], marker="o", color=c, linestyle="none", markersize=6.0,
                      markeredgecolor="black", markeredgewidth=0.4,
                      label=f"Y5 grid, $\\beta$={b:g}")
        for b, c in BETA_COLOR.items()
    ]
    theta_handles = [
        mlines.Line2D([], [], marker=THETA_SHAPE[t], color="#666666", linestyle="none",
                      markersize=6.0, markeredgecolor="black", markeredgewidth=0.4,
                      label=f"$\\theta$={t:g}" if t else "$\\theta$=0")
        for t in THETA_SHAPE
    ]
    other = [
        mlines.Line2D([], [], marker=RANDOM_SHAPE, color=RANDOM_COLOR, linestyle="none",
                      markersize=6.5, markeredgecolor="black", markeredgewidth=0.4,
                      label="Random@ret (Y4)"),
        mlines.Line2D([], [], marker=IS_SHAPE, color=IS_COLOR, linestyle="none",
                      markersize=8.0, markeredgecolor="black", markeredgewidth=0.5,
                      label="IS (Y1)"),
        mlines.Line2D([], [], marker=SPURS_SHAPE, color=SPURS_COLOR, linestyle="none",
                      markersize=13.0, markeredgecolor="black", markeredgewidth=0.5,
                      label="SPURS (Y1)"),
        mlines.Line2D([], [], color="#999999", lw=0.7, ls=(0, (4, 3)), label="Raw (Y1)"),
    ]
    fig.legend(handles=beta_handles + other, loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, 0.045), handletextpad=0.3, columnspacing=1.0)
    fig.legend(handles=theta_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.035), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    return fig


def main() -> None:
    data = load()
    _style.assert_no_ml()
    for variant, hard in (("", False), ("hard", True)):
        fig = draw(data, hard=hard)
        _style.save_fig(fig, NAME, variant=variant or None)
        plt.close(fig)
    _style.assert_no_ml()
    print(f"{NAME}: {len(data['sweep'])} sweep cells + 4 reference points "
          f"over {len(PANELS)} panels")


if __name__ == "__main__":
    main()
