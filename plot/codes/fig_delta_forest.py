#!/usr/bin/env python
"""Appendix figure: forest plot of every paired bootstrap comparison (App. G).

Reads only ``analysis/paper/out/bootstrap.csv`` (S1/S4 statistics stage).  No
model is trained, fitted, evaluated or selected here; the CIs are already in the
file.

One row per comparison in the statistical protocol:
  x = ``delta_hat`` (paired pooled-instance macro-F1 difference)
  horizontal bar = the two-sided 90% percentile CI (``ci_low`` .. ``ci_high``)
  shaded band = the non-inferiority margin ``delta`` (0.01), so a bar whose
      whole interval sits inside the band is the visual form of "within delta".
Rows are grouped by pre-committed gate (G-A .. G-H) and, last, the beta/theta
sensitivity contrasts that carry no gate; groups are ordered top to bottom and
each carries its own label.

Rows are coloured by ``equivalence_status``:
  * equivalent                  - the CI lies inside +/- delta
  * non-inferior, not equivalent - inside the upper margin only
  * not established              - otherwise

``torch`` / ``transformers`` are asserted absent by ``_style``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _style  # noqa: E402  (installs the no-torch guard)

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
import pandas as pd  # noqa: E402

NAME = "fig_delta_forest"

GATE_ORDER = ["G-A", "G-C", "G-D", "G-E", "G-F", "G-G", "G-H"]
GATE_LABEL = {
    "G-A": "G-A  SPURS vs Raw (per dataset)",
    "G-C": "G-C  schedule vs composite order",
    "G-D": "G-D  single-signal selectors",
    "G-E": "G-E  IS vs random subset at matched retention",
    "G-F": "G-F  SPURS vs reduced raw budget",
    "G-G": "G-G  factorial interaction",
    "G-H": "G-H  beta x theta sensitivity",
    "": "beta / theta contrasts (no gate)",
}
STATUS_COLOR = {
    "equivalent": "#0072B2",
    "non-inferior, not equivalent": "#E69F00",
    "not established": "#999999",
}


def load() -> pd.DataFrame:
    b = pd.read_csv(_style.OUT_DIR / "bootstrap.csv")
    b = b.dropna(subset=["delta_hat", "ci_low", "ci_high"]).copy()
    b["gate_key"] = b["gate"].fillna("").astype(str)
    b.loc[b["gate_key"].isin(["nan"]), "gate_key"] = ""
    order = {g: i for i, g in enumerate(GATE_ORDER + [""])}
    b["gorder"] = b["gate_key"].map(order)
    if b["gorder"].isna().any():
        bad = sorted(b.loc[b["gorder"].isna(), "gate_key"].unique())
        raise SystemExit(f"unmapped gate(s): {bad}")
    b = b.sort_values(["gorder", "delta_hat"], ascending=[True, False]).reset_index(drop=True)
    b["ypos"] = range(len(b))
    return b


def draw(b: pd.DataFrame) -> plt.Figure:
    _style.apply_style()
    n = len(b)
    height = max(3.2, 0.155 * n)
    fig, ax = plt.subplots(figsize=(6.6, height))

    delta = float(b["delta"].dropna().iloc[0])
    if b["delta"].dropna().nunique() != 1:
        raise SystemExit("more than one non-inferiority margin in bootstrap.csv")
    ax.axvspan(-delta, delta, color="#B2182B", alpha=0.08, zorder=0, lw=0)
    for x in (-delta, delta):
        ax.axvline(x, color="#B2182B", lw=0.7, ls=(0, (3, 2)), zorder=1)

    for _, r in b.iterrows():
        col = STATUS_COLOR.get(str(r["equivalence_status"]), "#999999")
        ax.plot([r["ci_low"], r["ci_high"]], [r["ypos"], r["ypos"]],
                color=col, lw=1.1, solid_capstyle="butt", zorder=3)
        ax.plot([r["ci_low"], r["ci_low"]], [r["ypos"] - 0.22, r["ypos"] + 0.22],
                color=col, lw=0.9, zorder=3)
        ax.plot([r["ci_high"], r["ci_high"]], [r["ypos"] - 0.22, r["ypos"] + 0.22],
                color=col, lw=0.9, zorder=3)
        ax.plot([r["delta_hat"]], [r["ypos"]], marker="o", color=col,
                markersize=3.2, markeredgecolor="black", markeredgewidth=0.3, zorder=4)

    ax.axvline(0.0, color="black", lw=0.8, zorder=2)

    ax.set_yticks(b["ypos"])
    ax.set_yticklabels(b["comparison"], fontsize=6.0)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel("$\\Delta$ macro-F1 (paired pooled-instance bootstrap, 90% CI)")
    ax.grid(True, which="major", axis="x")
    ax.margins(x=0.10)

    # Group separators and right-hand gate labels.
    xmax = float(b[["ci_low", "ci_high"]].abs().to_numpy().max())
    for gk, g in b.groupby("gate_key", sort=False):
        lo, hi = g["ypos"].min(), g["ypos"].max()
        if lo > 0:
            ax.axhline(lo - 0.5, color="#666666", lw=0.5, ls=(0, (2, 3)), zorder=2)
        ax.text(xmax * 1.02, (lo + hi) / 2.0, GATE_LABEL[gk], fontsize=6.5,
                va="center", ha="left", color="#333333")

    handles = [
        mlines.Line2D([], [], color=STATUS_COLOR[s], lw=1.4, marker="o",
                      markersize=3.2, markeredgecolor="black", markeredgewidth=0.3,
                      label=s)
        for s in ("equivalent", "non-inferior, not equivalent", "not established")
    ]
    handles.append(
        mlines.Line2D([], [], color="#B2182B", lw=4.0, alpha=0.25, label=f"$\\pm\\delta$ = {delta:g}")
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.008),
               handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.03, 0.80, 1))
    return fig


def main() -> None:
    b = load()
    _style.assert_no_ml()
    fig = draw(b)
    _style.save_fig(fig, NAME)
    plt.close(fig)
    _style.assert_no_ml()
    print(f"{NAME}: {len(b)} comparisons, {b['gate_key'].replace('', '(none)').nunique()} groups")


if __name__ == "__main__":
    main()
