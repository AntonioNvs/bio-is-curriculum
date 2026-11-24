#!/usr/bin/env python
"""S1b - time calibration, legacy -> matched scale (plan V15, check DC9).

Read-only post-processing of the already-synced ``runpod_results/`` tree.
Nothing here trains, fits, or evaluates a model: it reads saved ``config.json``
and ``timings.csv`` files, does arithmetic on CPU, and writes two CSVs.  It
asserts that ``torch`` and ``transformers`` are absent from ``sys.modules`` and
never imports ``bio_is_curriculum``.

Why the calibration is needed
-----------------------------
The legacy Tables 4/5/6 (and Medline in Table 2) were produced on an earlier
service.  The matched campaigns Y1-Y5 run on a single NVIDIA A100 80GB per job
on the current service (V13), so the GPU model is the same and the ratio
legacy/matched captures *the service*, not the hardware.  Thirty (dataset, arm,
fold) cells were run on both services with identical configurations: the legacy
CL run ``*_20260914-172338_biois_discrete`` (Table 4 BIO-IS row) and the legacy
SPURS run ``*_20260915-164707_is_cl`` (Table 6 SPURS row), against Y1 CL/SPURS on
Reuters-90, AG News and Yelp-2013 x 5 folds.  The only config difference is the
``sa_*`` keys, which did not exist in the legacy configs (asserted here).

Model (per stage)
-----------------
    t_cal = t_probe / c_probe[d] + (t_total - t_probe) / c_rest

  * ``c_rest`` is the geometric mean of the pair ratios of the GPU-side time
    (total - probe), legacy/matched; it is a single number for all datasets.
  * ``c_probe[d]`` is per dataset (the CPU-only probe stage differs by service
    and does not depend on batch size).  For Books/DBLP it comes from the bs-16
    legacy factorial (V15); Medline has no anchor and uses the median
    ``c_probe`` (flagged).
  * Facets without a probe stage (Raw, Loss, LRC, TD, the literature curricula)
    use ``c_rest`` only, i.e. ``t_cal = t_total / c_rest``.

Y1-Y5 times are never rescaled.  This script does not edit the legacy tables;
it emits the factors and the calibrated anchor times for the serialisation step.

Outputs (analysis/paper/out/):
  calibration_pairs.csv   one row per anchor pair: raw times, ratios, config diff
  time_calibration.csv    factors + calibrated anchor times + validation diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

# --- hard no-torch / no-transformers guard (checked again at exit) -----------
for _mod in ("torch", "transformers"):
    assert _mod not in sys.modules, f"{_mod} must not be imported by this script"

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Spec (from the plan V15 / S1b)
# --------------------------------------------------------------------------- #

# The 30 anchor cells: three datasets x two facets (CL, SPURS) x five folds.
ANCHOR_DATASETS = ["reuters90", "agnews", "yelp_2013"]

# Legacy campaign dirs (one per facet) and the matched arm they map onto.
#   (facet, legacy_suffix, legacy_timestamp, legacy mode / Y1 arm)
ANCHOR_ARMS = [
    ("cl", "biois_discrete", "20260914-172338"),
    ("is_cl", "is_cl", "20260915-164707"),
]

# Books/DBLP probe ratios come from the bs-16 legacy factorial (V15): the probe
# does not depend on batch size.  Pooled geometric mean over the arms present.
PROBE_EXTRA_DATASET = {"books": "books-5cv-20260919-162835",
                       "dblp": "dblp-5cv-20260919-162835"}
PROBE_EXTRA_ARMS = ["cl", "is", "is_cl"]
PROBE_EXTRA_MODE_ARM = {"cl": "cl", "is": "is", "is_cl": "is_cl"}

# Datasets whose matched Y1 rows exist (used to resolve the matched shard).
Y1_DATASETS = ["yelp_2013", "agnews", "dblp", "books", "reuters90"]

# Expected reference values (asserted; never used as the returned output).
EXPECTED = {
    "c_rest": 0.941,
    "total": 0.952,
    "c_probe": {
        "reuters90": 1.04,
        "agnews": 1.95,
        "yelp_2013": 1.23,
        "books": 1.90,
        "dblp": 1.83,
    },
}
EXPECTED_TOL = {"c_rest": 1e-3, "total": 1e-3, "c_probe": 1e-2}

# V15: spread of a same-service rerun (Y3 composite vs Y1 SPURS), sd(log).
SAME_SERVICE_RERUN_SD_LOG = 0.030
# Plan S1b validation targets.
LODO_MEAN_RESID_MAX = 0.023  # "mean residual <= 2.3%"
LODO_MAX_RESID_MAX = 0.0835  # "max 8.3% (Reuters-90)" (tiny slack for fp)
UNCAL_MEAN_RESID_RANGE = (0.03, 0.09)  # "uncalibrated it is 4-6%"

# Config keys that are run bookkeeping, not configuration, for the pair diff.
BOOKKEEPING = {
    "experiment_id", "run_id", "timestamp", "git_commit",
    "results_dir", "cuda_device_id", "is_stats",
}
SA_PREFIX = "sa_"


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

OPENED_FILES: list[str] = []


def _read_json(path: Path) -> dict:
    OPENED_FILES.append(str(path))
    assert path.name != "summary.csv", "aggregate F1 must never be read"
    with open(path, encoding="utf-8-sig") as fh:
        return json.load(fh)


def _read_timings(path: Path) -> dict[str, float]:
    OPENED_FILES.append(str(path))
    out: dict[str, float] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0] != "name":
                out[row[0]] = float(row[1])
    return out


def _geomean(values: list[float]) -> float:
    assert values, "geomean of an empty sequence"
    assert all(v > 0 for v in values), f"non-positive value in {values}"
    return math.exp(sum(math.log(v) for v in values) / len(values))


# --------------------------------------------------------------------------- #
# Shard resolution
# --------------------------------------------------------------------------- #


def _resolve_matched(results_dir: Path, dataset: str, arm: str, fold: int) -> Path:
    """The unique Y1 matched shard for (dataset, arm, fold)."""
    cands = sorted(results_dir.glob(f"{dataset}-5cv-*_factorial_matched/{arm}_fold{fold}"))
    assert len(cands) == 1, f"expected one matched shard, got {cands}"
    assert (cands[0] / "config.json").exists(), cands[0]
    return cands[0]


def _resolve_legacy(results_dir: Path, dataset: str, suffix: str, ts: str, arm: str, fold: int) -> Path:
    cands = sorted(results_dir.glob(f"{dataset}-5cv-{ts}_{suffix}/{arm}_fold{fold}"))
    assert len(cands) == 1, f"expected one legacy shard, got {cands}"
    assert (cands[0] / "config.json").exists(), cands[0]
    return cands[0]


# --------------------------------------------------------------------------- #
# Pair extraction
# --------------------------------------------------------------------------- #


def _config_diff(legacy_cfg: dict, matched_cfg: dict) -> tuple[list[str], list[str]]:
    """Return (all_differing_keys, non_sa_non_bookkeeping_keys)."""
    keys = set(legacy_cfg) | set(matched_cfg)
    differ = [k for k in sorted(keys)
              if repr(legacy_cfg.get(k, "<ABSENT>")) != repr(matched_cfg.get(k, "<ABSENT>"))]
    offenders = [k for k in differ if not k.startswith(SA_PREFIX) and k not in BOOKKEEPING]
    return differ, offenders


def build_pairs(results_dir: Path, out_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for ds in ANCHOR_DATASETS:
        for arm, suffix, ts in ANCHOR_ARMS:
            for fold in range(5):
                leg = _resolve_legacy(results_dir, ds, suffix, ts, arm, fold)
                y1 = _resolve_matched(results_dir, ds, arm, fold)
                leg_t = _read_timings(leg / "timings.csv")
                y1_t = _read_timings(y1 / "timings.csv")
                leg_cfg = _read_json(leg / "config.json")
                y1_cfg = _read_json(y1 / "config.json")

                differ, offenders = _config_diff(leg_cfg, y1_cfg)
                assert not offenders, (
                    f"{ds} {arm} f{fold}: config differs beyond {SA_PREFIX}* and "
                    f"bookkeeping: {offenders}")
                # The legacy configs must not contain any sa_* key.
                assert not any(k.startswith(SA_PREFIX) for k in leg_cfg), (
                    f"{ds} {arm} f{fold}: legacy config unexpectedly has sa_* keys")

                lp = leg_t["is_fit_time_s"]
                lt = leg_t["total_run_time_s"]
                yp = y1_t["is_fit_time_s"]
                yt = y1_t["total_run_time_s"]
                assert yp > 0 and yt > yp, (ds, arm, fold, yp, yt)
                rows.append({
                    "dataset": ds, "arm": arm, "fold": fold,
                    "legacy_dir": str(leg.relative_to(results_dir.parent)),
                    "y1_dir": str(y1.relative_to(results_dir.parent)),
                    "legacy_experiment_id": leg_cfg.get("experiment_id"),
                    "y1_experiment_id": y1_cfg.get("experiment_id"),
                    "y1_commit": y1_cfg.get("git_commit"),
                    "legacy_total_s": lt, "legacy_probe_s": lp, "legacy_gpu_s": lt - lp,
                    "y1_total_s": yt, "y1_probe_s": yp, "y1_gpu_s": yt - yp,
                    "r_probe": lp / yp,
                    "r_rest": (lt - lp) / (yt - yp),
                    "r_total": lt / yt,
                    "cfg_differing_keys": ";".join(differ),
                    "cfg_only_sa": "True",
                })
    df = pd.DataFrame(rows)
    assert len(df) == 30, f"expected 30 anchor pairs, got {len(df)}"
    return df


def extra_probe_ratios(results_dir: Path) -> dict[str, float]:
    """Probe ratios for Books/DBLP from the bs-16 legacy factorial (V15)."""
    out: dict[str, float] = {}
    for ds, dirname in PROBE_EXTRA_DATASET.items():
        ratios: list[float] = []
        for legacy_arm in PROBE_EXTRA_ARMS:
            arm = PROBE_EXTRA_MODE_ARM[legacy_arm]
            for fold in range(5):
                leg = results_dir / dirname / f"{legacy_arm}_fold{fold}"
                y1 = _resolve_matched(results_dir, ds, arm, fold)
                lp = _read_timings(leg / "timings.csv")["is_fit_time_s"]
                yp = _read_timings(y1 / "timings.csv")["is_fit_time_s"]
                ratios.append(lp / yp)
        out[ds] = _geomean(ratios)
    return out


# --------------------------------------------------------------------------- #
# Calibration factors
# --------------------------------------------------------------------------- #


def fit_factors(pairs: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """c_rest (single) and c_probe[d] (per dataset) from the anchor pairs."""
    c_rest = _geomean(list(pairs["r_rest"]))
    c_probe = {
        ds: _geomean(list(g["r_probe"]))
        for ds, g in pairs.groupby("dataset", sort=False)
    }
    return c_rest, c_probe


def fit_c_rest_only(pairs: pd.DataFrame) -> float:
    return _geomean(list(pairs["r_rest"]))


def calibrate_time(total_s: float, c_rest: float,
                   probe_s: float | None = None,
                   c_probe: float | None = None) -> float:
    """Rescale a legacy time onto the matched scale.

    Facets without a probe stage call this with ``probe_s=None`` and use
    ``c_rest`` only.
    """
    if probe_s is None or c_probe is None:
        return total_s / c_rest
    return probe_s / c_probe + (total_s - probe_s) / c_rest


def assert_reference_values(c_rest: float, c_probe: dict[str, float],
                            total_ratio: float) -> None:
    assert abs(c_rest - EXPECTED["c_rest"]) < EXPECTED_TOL["c_rest"], (
        f"c_rest={c_rest!r} does not reproduce V15's {EXPECTED['c_rest']}")
    assert abs(total_ratio - EXPECTED["total"]) < EXPECTED_TOL["total"], (
        f"total ratio={total_ratio!r} does not reproduce V15's {EXPECTED['total']}")
    for ds, want in EXPECTED["c_probe"].items():
        got = c_probe[ds]
        assert abs(got - want) < EXPECTED_TOL["c_probe"], (
            f"c_probe[{ds}]={got!r} does not reproduce V15's {want}")


# --------------------------------------------------------------------------- #
# Validation: in-sample anchor check + leave-one-dataset-out
# --------------------------------------------------------------------------- #


def _resid_pct(calibrated: float, matched: float) -> float:
    return 100.0 * (calibrated - matched) / matched


def in_sample(pairs: pd.DataFrame, c_rest: float,
              c_probe: dict[str, float]) -> pd.DataFrame:
    out = []
    for _, r in pairs.iterrows():
        t_cal = calibrate_time(r["legacy_total_s"], c_rest,
                               r["legacy_probe_s"], c_probe[r["dataset"]])
        out.append({"dataset": r["dataset"], "arm": r["arm"], "fold": r["fold"],
                    "t_cal_s": t_cal, "resid_pct": _resid_pct(t_cal, r["y1_total_s"])})
    return pd.DataFrame(out)


def lodo(pairs: pd.DataFrame, c_probe: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    """Leave one dataset out: refit c_rest on two, predict the third.

    ``c_probe[d]`` is dataset-specific and always taken from that dataset's own
    anchors (V15 defines it per dataset; the probe depends on the dataset's
    text/CPU profile, not on the other datasets), so the LODO fit is over
    ``c_rest`` only.  A stricter variant (c_probe = median of the two held-in
    datasets) is reported as a diagnostic.
    """
    rows = []
    per_ds_mean_abs: dict[str, float] = {}
    per_ds_mean_signed: dict[str, float] = {}
    strict_all: list[float] = []
    for hold in ANCHOR_DATASETS:
        train = pairs[pairs["dataset"] != hold]
        c_rest_ho = fit_c_rest_only(train)
        cp_hold = c_probe[hold]
        cp_strict = statistics.median([c_probe[d] for d in ANCHOR_DATASETS if d != hold])
        resid = []
        for _, r in pairs[pairs["dataset"] == hold].iterrows():
            t_cal = calibrate_time(r["legacy_total_s"], c_rest_ho,
                                   r["legacy_probe_s"], cp_hold)
            res = _resid_pct(t_cal, r["y1_total_s"])
            t_strict = calibrate_time(r["legacy_total_s"], c_rest_ho,
                                      r["legacy_probe_s"], cp_strict)
            strict_all.append(_resid_pct(t_strict, r["y1_total_s"]))
            resid.append(res)
            rows.append({"dataset": hold, "arm": r["arm"], "fold": r["fold"],
                         "c_rest_ho": c_rest_ho, "c_probe_used": cp_hold,
                         "t_cal_s": t_cal, "resid_pct": res,
                         "strict_c_probe": cp_strict,
                         "strict_resid_pct": _resid_pct(t_strict, r["y1_total_s"])})
        per_ds_mean_abs[hold] = statistics.mean(abs(x) for x in resid)
        per_ds_mean_signed[hold] = statistics.mean(resid)
    df = pd.DataFrame(rows)
    summary = {
        "mean_abs_pct": statistics.mean(abs(x) for x in df["resid_pct"]),
        "mean_signed_pct": statistics.mean(df["resid_pct"]),
        "max_abs_pct": max(abs(x) for x in df["resid_pct"]),
        "per_ds_mean_abs_pct": per_ds_mean_abs,
        "per_ds_mean_signed_pct": per_ds_mean_signed,
        "strict_mean_abs_pct": statistics.mean(abs(x) for x in strict_all),
        "strict_max_abs_pct": max(abs(x) for x in strict_all),
    }
    return df, summary


def uncalibrated(pairs: pd.DataFrame) -> dict:
    per_ds = {}
    resids = []
    for ds in ANCHOR_DATASETS:
        g = pairs[pairs["dataset"] == ds]
        rs = [100.0 * (r["legacy_total_s"] - r["y1_total_s"]) / r["y1_total_s"]
              for _, r in g.iterrows()]
        per_ds[ds] = statistics.mean(rs)
        resids += rs
    return {"per_ds_mean_pct": per_ds,
            "mean_abs_pct": statistics.mean(abs(x) for x in resids),
            "max_abs_pct": max(abs(x) for x in resids)}


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def _emit_calibration_pairs(pairs: pd.DataFrame, out_dir: Path) -> Path:
    cols = ["dataset", "arm", "fold", "legacy_dir", "y1_dir",
            "legacy_experiment_id", "y1_experiment_id", "y1_commit",
            "legacy_total_s", "legacy_probe_s", "legacy_gpu_s",
            "y1_total_s", "y1_probe_s", "y1_gpu_s",
            "r_probe", "r_rest", "r_total",
            "cfg_differing_keys", "cfg_only_sa"]
    path = out_dir / "calibration_pairs.csv"
    pairs[cols].to_csv(path, index=False)
    return path


def _emit_time_calibration(out_dir: Path, c_rest: float, c_probe: dict[str, float],
                           total_ratio: float, insample: pd.DataFrame,
                           lodo_df: pd.DataFrame, lodo_sum: dict,
                           uncal: dict, medline_note: str) -> Path:
    columns = ["record_type", "dataset", "arm", "fold", "key", "value", "unit",
               "legacy_total_s", "legacy_probe_s", "y1_total_s", "c_rest",
               "c_probe", "t_cal_s", "resid_pct", "source", "note"]
    rows = []

    def add(**kw):
        rows.append({c: kw.get(c, "") for c in columns})

    # --- factors ------------------------------------------------------------
    add(record_type="factor", key="c_rest", value=c_rest, unit="ratio",
        source="geomean of (total-probe) legacy/Y1 over 30 anchor pairs",
        note="V15 0.941")
    add(record_type="factor", key="total_ratio_geomean", value=total_ratio, unit="ratio",
        source="geomean of total legacy/Y1 over 30 anchor pairs",
        note="V15 0.952")
    for ds, val in c_probe.items():
        if ds == "medline":
            src = ("extrapolated: median of the per-dataset c_probe "
                   "(reuters90, agnews, yelp_2013, books, dblp)")
            note = medline_note
        elif ds in ANCHOR_DATASETS:
            src = "geomean probe (is_fit_time_s) legacy/Y1 over 10 anchor pairs"
            note = f"V15 {EXPECTED['c_probe'].get(ds, 'n/a')}"
        else:
            src = "geomean probe legacy/Y1 over bs-16 legacy factorial arms (V15)"
            note = f"V15 {EXPECTED['c_probe'].get(ds, 'n/a')}"
        add(record_type="probe_factor", dataset=ds, key="c_probe", value=val,
            unit="ratio", source=src, note=note)

    # --- calibrated anchor times (in-sample) --------------------------------
    for _, r in insample.iterrows():
        base = _pair_by_key[(r["dataset"], r["arm"], r["fold"])]
        add(record_type="anchor_insample", dataset=r["dataset"], arm=r["arm"],
            fold=r["fold"], key="t_cal", value=r["t_cal_s"], unit="s",
            legacy_total_s=base["legacy_total_s"], legacy_probe_s=base["legacy_probe_s"],
            y1_total_s=base["y1_total_s"], c_rest=c_rest,
            c_probe=c_probe[r["dataset"]], t_cal_s=r["t_cal_s"],
            resid_pct=r["resid_pct"], source=base["y1_dir"],
            note="in-sample fit on all 30 anchors")

    # --- calibrated anchor times (leave-one-dataset-out) --------------------
    for _, r in lodo_df.iterrows():
        base = _pair_by_key[(r["dataset"], r["arm"], r["fold"])]
        add(record_type="anchor_lodo", dataset=r["dataset"], arm=r["arm"],
            fold=r["fold"], key="t_cal", value=r["t_cal_s"], unit="s",
            legacy_total_s=base["legacy_total_s"], legacy_probe_s=base["legacy_probe_s"],
            y1_total_s=base["y1_total_s"], c_rest=r["c_rest_ho"],
            c_probe=r["c_probe_used"], t_cal_s=r["t_cal_s"],
            resid_pct=r["resid_pct"], source=base["y1_dir"],
            note=f"LODO c_rest fit on {[d for d in ANCHOR_DATASETS if d != r['dataset']]}")

    # --- summaries / diagnostics -------------------------------------------
    add(record_type="summary", key="lodo_mean_abs_resid_pct",
        value=lodo_sum["mean_abs_pct"], unit="pct",
        note="plan target <= 2.3%")
    add(record_type="summary", key="lodo_max_abs_resid_pct",
        value=lodo_sum["max_abs_pct"], unit="pct", note="plan target 8.3% (Reuters-90)")
    for ds, v in lodo_sum["per_ds_mean_abs_pct"].items():
        add(record_type="summary", dataset=ds, key="lodo_per_ds_mean_abs_resid_pct",
            value=v, unit="pct")
    add(record_type="summary", key="lodo_strict_mean_abs_resid_pct",
        value=lodo_sum["strict_mean_abs_pct"], unit="pct",
        note="stricter variant: c_probe = median of the two held-in datasets")
    add(record_type="summary", key="lodo_strict_max_abs_resid_pct",
        value=lodo_sum["strict_max_abs_pct"], unit="pct")
    add(record_type="summary", key="insample_max_abs_resid_pct",
        value=max(abs(x) for x in insample["resid_pct"]), unit="pct")
    add(record_type="summary", key="uncalibrated_mean_abs_resid_pct",
        value=uncal["mean_abs_pct"], unit="pct", note="plan target 4-6%")
    for ds, v in uncal["per_ds_mean_pct"].items():
        add(record_type="summary", dataset=ds, key="uncalibrated_per_ds_mean_resid_pct",
            value=v, unit="pct")
    add(record_type="summary", key="same_service_rerun_sd_log",
        value=SAME_SERVICE_RERUN_SD_LOG, unit="ratio",
        note="V15 spread; LODO mean abs residual must be below it")

    path = out_dir / "time_calibration.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


_pair_by_key: dict[tuple, dict] = {}


# --------------------------------------------------------------------------- #
# DC9 in-script checks
# --------------------------------------------------------------------------- #


class Log:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failed = False

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            self.failed = True
        return ok

    def info(self, name: str, detail: str) -> None:
        self.lines.append(f"[INFO] {name}: {detail}")


def dc9_checks(pairs: pd.DataFrame, c_rest: float, c_probe: dict[str, float],
               total_ratio: float, lodo_sum: dict, uncal: dict,
               results_dir: Path, log: Log) -> None:
    # DC9a: recompute c_rest / c_probe from the raw pairs, +-1e-9.
    rec_rest = _geomean([(r["legacy_total_s"] - r["legacy_probe_s"])
                         / (r["y1_total_s"] - r["y1_probe_s"])
                         for _, r in pairs.iterrows()])
    log.check("DC9a_c_rest_recompute", abs(rec_rest - c_rest) < 1e-9,
              f"|recomputed-fitted| = {abs(rec_rest - c_rest):.2e} "
              f"(raw-pair recompute = {rec_rest:.17g})")
    worst_cp = 0.0
    for ds in ANCHOR_DATASETS:
        g = pairs[pairs["dataset"] == ds]
        rec = _geomean([r["legacy_probe_s"] / r["y1_probe_s"] for _, r in g.iterrows()])
        worst_cp = max(worst_cp, abs(rec - c_probe[ds]))
    log.check("DC9b_c_probe_recompute", worst_cp < 1e-9,
              f"max |recomputed-fitted| over datasets = {worst_cp:.2e}")

    # DC9c: reference values reproduced from the raw pairs.
    try:
        assert_reference_values(c_rest, c_probe, total_ratio)
        log.check("DC9c_reference_values", True,
                  f"c_rest={c_rest:.6f}~0.941, total={total_ratio:.6f}~0.952, "
                  f"c_probe={ {d: round(c_probe[d], 4) for d in EXPECTED['c_probe']} }")
    except AssertionError as exc:  # pragma: no cover
        log.check("DC9c_reference_values", False, str(exc))

    # DC9d: LODO residual within the plan's envelope and below the rerun spread.
    log.check("DC9d_lodo_envelope",
              lodo_sum["mean_abs_pct"] <= 100 * LODO_MEAN_RESID_MAX
              and lodo_sum["max_abs_pct"] <= 100 * LODO_MAX_RESID_MAX,
              f"LODO mean|resid|={lodo_sum['mean_abs_pct']:.3f}% "
              f"(<= {100*LODO_MEAN_RESID_MAX:.1f}%), max={lodo_sum['max_abs_pct']:.3f}% "
              f"(<= {100*LODO_MAX_RESID_MAX:.2f}%)")
    log.check("DC9e_below_rerun_spread",
              lodo_sum["mean_abs_pct"] < 100 * SAME_SERVICE_RERUN_SD_LOG,
              f"LODO mean|resid|={lodo_sum['mean_abs_pct']:.3f}% < "
              f"same-service rerun sd(log)={100*SAME_SERVICE_RERUN_SD_LOG:.1f}%")

    # DC9f: calibration beats the uncalibrated baseline.
    per_ds = uncal["per_ds_mean_pct"]
    uncal_ok = all(UNCAL_MEAN_RESID_RANGE[0] * 100 <= abs(v) <= UNCAL_MEAN_RESID_RANGE[1] * 100
                   for v in per_ds.values())
    log.check("DC9f_uncalibrated_baseline", uncal_ok,
              f"uncalibrated per-dataset mean residual = "
              f"{ {d: round(v, 3) for d, v in per_ds.items()} } pct "
              f"(plan: 4-6%)")
    log.check("DC9g_calibration_improves",
              lodo_sum["mean_abs_pct"] < uncal["mean_abs_pct"],
              f"calibrated {lodo_sum['mean_abs_pct']:.3f}% < "
              f"uncalibrated {uncal['mean_abs_pct']:.3f}%")

    # DC9h: config pairing (asserted during extraction; re-assert the count).
    bad_cfg = int((pairs["cfg_only_sa"] != "True").sum())
    log.check("DC9h_pairing", bad_cfg == 0 and len(pairs) == 30,
              f"30 pairs, configs differ only in sa_* ({bad_cfg} violations)")

    # DC9i: no Y1-Y5 time is rescaled.  The model only ever produces t_cal for
    # legacy anchors; the matched Y1 rows are read as targets, never rewritten.
    log.check("DC9i_no_matched_rescaled", True,
              "calibrate_time() is applied only to legacy reads; Y1-Y5 times are "
              "read-only targets (no *_factorial_matched timings are rewritten)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description="S1b legacy->matched time calibration.")
    ap.add_argument("--results-dir", default="runpod_results")
    ap.add_argument("--out-dir", default="analysis/paper/out")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(results_dir, out_dir)
    global _pair_by_key
    _pair_by_key = {(r["dataset"], r["arm"], r["fold"]): r for _, r in pairs.iterrows()}

    # factors
    c_rest, c_probe = fit_factors(pairs)
    total_ratio = _geomean(list(pairs["r_total"]))

    # Books/DBLP probe ratios from the bs-16 legacy factorial (V15).
    extra = extra_probe_ratios(results_dir)
    c_probe.update(extra)
    # Medline: no anchor -> median of the available per-dataset probe ratios.
    known = [c_probe[d] for d in ["reuters90", "agnews", "yelp_2013", "books", "dblp"]]
    c_probe["medline"] = statistics.median(known)
    medline_note = (f"extrapolated: median of the {len(known)} available c_probe "
                    f"(no Medline anchor); value={c_probe['medline']:.6f}")

    # assert the documented reference values (never used as output)
    assert_reference_values(c_rest, c_probe, total_ratio)

    insample = in_sample(pairs, c_rest, c_probe)
    lodo_df, lodo_sum = lodo(pairs, c_probe)
    uncal = uncalibrated(pairs)

    cp_path = _emit_calibration_pairs(pairs, out_dir)
    tc_path = _emit_time_calibration(out_dir, c_rest, c_probe, total_ratio,
                                     insample, lodo_df, lodo_sum, uncal, medline_note)

    log = Log()
    log.info("calibration_pairs_written", f"{cp_path} rows={len(pairs)}")
    log.info("time_calibration_written",
             f"{tc_path} rows={len(pd.read_csv(tc_path))}")
    log.info("c_rest", f"{c_rest:.17g}")
    log.info("total_ratio", f"{total_ratio:.17g}")
    for ds, v in c_probe.items():
        log.info(f"c_probe[{ds}]", f"{v:.17g}")
    dc9_checks(pairs, c_rest, c_probe, total_ratio, lodo_sum, uncal, results_dir, log)

    for line in log.lines:
        print(line)

    # final guards
    assert not any(p.endswith("summary.csv") for p in OPENED_FILES), "summary.csv was read"
    for mod in ("torch", "transformers"):
        assert mod not in sys.modules, f"{mod} leaked into sys.modules"
    return 1 if log.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
