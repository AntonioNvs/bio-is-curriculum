#!/usr/bin/env python
"""Appendix figure: where the time per fold actually goes, by arm.

Reads only ``analysis/paper/out/runs_index.csv`` (S1).  Nothing here trains,
fits, evaluates or selects.

The cost columns already partition ``total_run_time_s`` exactly, so the stacked
bar is a decomposition, not an estimate.  ``cost_eval`` is split into the final
evaluation every arm pays and the in-loop non-final evaluations that only the
curriculum arms pay (V9) - the overhead the ``__adjusted`` Pareto variant
removes.  Drawing it as its own hatched segment makes that correction legible
instead of a footnote.

Panels: AG News, Books, Reuters-90 (the same three as the Pareto figure).
  y = arm, x = fold-mean seconds per fold, segments summing to the total.

``torch`` / ``transformers`` are asserted absent by ``_style``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _style  # noqa: E402  (installs the no-torch guard)

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import pandas as pd  # noqa: E402

NAME = "fig_cost_breakdown"
PANELS = ["agnews", "books", "reuters90"]

# (label, colour, hatch) - drawn bottom to top.
SEGMENTS = [
    ("Data load", "#999999", ""),
    ("Probe + select", "#56B4E9", ""),
    ("Fine-tune", "#0072B2", ""),
    ("Final eval", "#E69F00", ""),
    ("In-loop eval (CL/SPURS only)", "#D55E00", "///"),
]

# (label, campaign, arm-regex) - same endpoints as the Pareto figure.
ENDPOINTS = [
    ("Raw@2", "Y2", r"^raw_ep2$"),
    ("Raw@4", "Y2", r"^raw_ep4$"),
    ("Raw@6", "Y1", r"^raw$"),
    ("IS", "Y1", r"^is$"),
    ("CL", "Y1", r"^cl$"),
    ("SPURS", "Y1", r"^is_cl$"),
    ("Random@ret", "Y4", r"^raw_rand"),
]


def load() -> pd.DataFrame:
    idx = pd.read_csv(_style.OUT_DIR / "runs_index.csv")
    rows = []
    for ds in PANELS:
        for label, campaign, arm_re in ENDPOINTS:
            sub = idx[(idx.campaign == campaign) & (idx.dataset == ds)
                      & idx.arm.str.match(arm_re)]
            if sub.empty:
                raise SystemExit(f"MISSING {label} for {ds} ({campaign} ~ {arm_re})")
            load_s = float(sub["cost_load"].mean())
            probe_s = float(sub["cost_probe_sel"].mean())
            finetune_s = float(sub["cost_finetune"].mean())
            eval_s = float(sub["cost_eval"].mean())
            nonfinal_s = float(sub["nonfinal_pred_time_s"].mean())
            total_s = float(sub["total_run_time_s"].mean())
            parts = [load_s, probe_s, finetune_s, eval_s - nonfinal_s, nonfinal_s]
            # Decomposition must reproduce the logged total.
            if abs(sum(parts) - total_s) > max(0.05, 1e-3 * total_s):
                raise SystemExit(
                    f"cost segments for {label}/{ds} sum to {sum(parts):.3f} "
                    f"but total_run_time_s is {total_s:.3f}"
                )
            rows.append({
                "dataset": ds, "series": label, "n_folds": len(sub),
                "load": load_s, "probe": probe_s, "finetune": finetune_s,
                "final_eval": eval_s - nonfinal_s, "inloop_eval": nonfinal_s,
                "total": total_s,
            })
    return pd.DataFrame(rows)


def draw(pts: pd.DataFrame) -> plt.Figure:
    _style.apply_style()
    labels = [e[0] for e in ENDPOINTS]
    ypos = {lab: i for i, lab in enumerate(reversed(labels))}  # Raw@2 on top
    keys = ["load", "probe", "finetune", "final_eval", "inloop_eval"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for ax, ds in zip(axes, PANELS):
        sub = pts[pts.dataset == ds]
        for _, r in sub.iterrows():
            y = ypos[r["series"]]
            left = 0.0
            for (lab, colour, hatch), k in zip(SEGMENTS, keys):
                w = float(r[k])
                if w <= 0:
                    continue
                ax.barh(y, w, left=left, height=0.72, color=colour, hatch=hatch,
                        edgecolor="white", linewidth=0.4, zorder=3)
                left += w
            ax.text(left * 1.02, y, f"{left:,.0f}", va="center", ha="left",
                    fontsize=6.0, color="#333333")
        ax.set_title(_style.DATASET_LABEL[ds], pad=3)
        ax.set_xlabel("Seconds per fold")
        ax.grid(True, which="major", axis="x", zorder=0)
        ax.margins(x=0.16)

    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels(list(ypos.keys()))
    axes[0].set_ylim(-0.6, len(labels) - 0.4)

    handles = [mpatches.Patch(facecolor=c, edgecolor="white", hatch=h, label=l)
               for l, c, h in SEGMENTS]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.4, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def main() -> None:
    pts = load()
    _style.assert_no_ml()
    fig = draw(pts)
    _style.save_fig(fig, NAME)
    plt.close(fig)
    _style.assert_no_ml()
    print(f"{NAME}: {len(pts)} arm x dataset bars over {len(PANELS)} panels")


if __name__ == "__main__":
    main()
