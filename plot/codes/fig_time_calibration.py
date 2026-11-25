#!/usr/bin/env python
"""S5 figure: legacy -> matched time calibration, one point per anchor fold.

Reads only the validated S1b outputs
``analysis/paper/out/calibration_pairs.csv`` (the 30 anchor pairs) and
``analysis/paper/out/time_calibration.csv`` (the fitted factors c_rest and the
per-dataset c_probe).

Main figure (log-log scatter):
  x = Y1 (matched-service) time, y = legacy time, one point per (dataset, arm,
  fold).  Colour = dataset, marker = arm (CL triangle, SPURS star).  The solid
  black line is the fitted ``c_rest`` model (y = c_rest * x); the dashed grey
  line is the identity y = x.

Variant ``__by_stage`` (1x2 panels): the same anchors split into the GPU-side
stage (fine-tuning/selection, scaled by the single ``c_rest``) and the probe
stage (CPU-only, scaled by the per-dataset ``c_probe``); the probe panel draws
one fitted line per dataset.

Nothing here trains, fits, evaluates or selects.  ``torch`` / ``transformers``
are asserted absent by ``_style``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _style  # noqa: E402  (installs the no-torch guard)

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NAME = "fig_time_calibration"
DATASETS = ["reuters90", "agnews", "yelp_2013"]  # anchor datasets
ARMS = ["cl", "is_cl"]


def load() -> tuple[pd.DataFrame, float, dict[str, float]]:
    pairs = pd.read_csv(_style.OUT_DIR / "calibration_pairs.csv")
    tc = pd.read_csv(_style.OUT_DIR / "time_calibration.csv")
    c_rest = float(tc.loc[tc["key"] == "c_rest", "value"].iloc[0])
    c_probe = {
        r["dataset"]: float(r["value"])
        for _, r in tc[tc["record_type"] == "probe_factor"].iterrows()
    }
    assert len(pairs) == 30, f"expected 30 anchor pairs, got {len(pairs)}"
    assert set(pairs["dataset"]) == set(DATASETS), sorted(set(pairs["dataset"]))
    assert set(pairs["arm"]) == set(ARMS), sorted(set(pairs["arm"]))
    return pairs, c_rest, c_probe


def _limits(a: pd.Series, b: pd.Series, pad: float = 0.12):
    """Log-symmetric limits covering both series with ``pad`` decades of room."""
    lo = min(float(a.min()), float(b.min()))
    hi = max(float(a.max()), float(b.max()))
    return 10 ** (np.log10(lo) - pad), 10 ** (np.log10(hi) + pad)


def _scatter(ax, pairs, xcol, ycol, label_axes=True):
    for ds in DATASETS:
        for arm in ARMS:
            sub = pairs[(pairs.dataset == ds) & (pairs.arm == arm)]
            if sub.empty:
                continue
            ax.scatter(sub[xcol], sub[ycol],
                       marker=_style.DATASET_MARKER[arm],
                       color=_style.DATASET_COLOR[ds],
                       edgecolors="black", linewidths=0.4,
                       s=42 if arm == "is_cl" else 26, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo, hi = _limits(pairs[xcol], pairs[ycol])
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.7, ls=(0, (4, 3)),
            zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    # Plain integer tick labels; matplotlib's log formatter uses a shared
    # exponent that collides on these narrow ranges.
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1, 3),
                                                  numticks=5))
        axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        axis.set_minor_formatter(mticker.NullFormatter())
    if label_axes:
        ax.set_xlabel("Y1 time (s, log)")
        ax.set_ylabel("Legacy time (s, log)")
    return lo, hi


def draw_main(pairs, c_rest) -> plt.Figure:
    _style.apply_style()
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    lo, hi = _scatter(ax, pairs, "y1_total_s", "legacy_total_s")
    ax.plot([lo, hi], [lo * c_rest, hi * c_rest], color="black", lw=1.0,
            zorder=2)
    ax.set_title("Total run time per fold")
    leg1 = ax.legend(handles=_dataset_handles(), loc="upper left", fontsize=7,
                     handletextpad=0.3, labelspacing=0.3)
    ax.add_artist(leg1)
    ax.legend(handles=_line_handles(c_rest), loc="lower right", fontsize=7,
              handletextpad=0.3, labelspacing=0.3)
    fig.tight_layout()
    return fig


def draw_by_stage(pairs, c_rest, c_probe) -> plt.Figure:
    _style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2))

    ax = axes[0]
    lo, hi = _scatter(ax, pairs, "y1_gpu_s", "legacy_gpu_s")
    ax.plot([lo, hi], [lo * c_rest, hi * c_rest], color="black", lw=1.0, zorder=2)
    ax.set_title("GPU-side stage (scaled by $c_{rest}$)")
    ax.legend(handles=_line_handles(c_rest), loc="upper left", fontsize=7,
              handletextpad=0.3, labelspacing=0.3)

    ax = axes[1]
    lo, hi = _scatter(ax, pairs, "y1_probe_s", "legacy_probe_s")
    for ds in DATASETS:
        cp = c_probe[ds]
        ax.plot([lo, hi], [lo * cp, hi * cp], color=_style.DATASET_COLOR[ds],
                lw=0.9, ls=(0, (1, 2)), zorder=2)
    ax.set_title("Probe stage ($c_{probe}$ per dataset)")
    # The probe panel is where the dataset colour matters (the $c_{probe}$
    # lines are per dataset); both panels share the arm/line encoding.
    ax.legend(handles=_dataset_handles() + _line_handles(c_probe=True),
              loc="upper left", fontsize=7, handletextpad=0.3, labelspacing=0.3)

    fig.tight_layout(w_pad=2.5)
    return fig


def _dataset_handles() -> list:
    ds = [mlines.Line2D([], [], marker="o", linestyle="none",
                        color=_style.DATASET_COLOR[d], markeredgecolor="black",
                        markeredgewidth=0.4, markersize=5,
                        label=_style.DATASET_LABEL[d]) for d in DATASETS]
    arms = [mlines.Line2D([], [], marker=_style.DATASET_MARKER[a],
                          linestyle="none", color="white",
                          markeredgecolor="black", markeredgewidth=0.5,
                          markersize=6, label=_style.ARM_LABEL[a]) for a in ARMS]
    return ds + arms


def _line_handles(c_rest=None, c_probe=False) -> list:
    h = [mlines.Line2D([], [], color="#999999", lw=0.7, ls=(0, (4, 3)),
                       label="identity")]
    if c_rest is not None:
        h.append(mlines.Line2D([], [], color="black", lw=1.0,
                               label=f"$c_{{rest}}={c_rest:.3f}$"))
    if c_probe:
        h.append(mlines.Line2D([], [], color="black", lw=0.9, ls=(0, (1, 2)),
                               label="$c_{probe}$"))
    return h


def main() -> None:
    pairs, c_rest, c_probe = load()
    _style.assert_no_ml()

    fig = draw_main(pairs, c_rest)
    _style.save_fig(fig, NAME)
    plt.close(fig)

    fig = draw_by_stage(pairs, c_rest, c_probe)
    _style.save_fig(fig, NAME, variant="by_stage")
    plt.close(fig)
    _style.assert_no_ml()
    print(f"{NAME}: {len(pairs)} anchor folds, c_rest={c_rest:.6f}")


if __name__ == "__main__":
    main()
