#!/usr/bin/env python
"""S4 - render the existing LaTeX table skeletons from the validated S1/S1b/S2 outputs.

Read-only post-processing of files that already exist on disk:
  * ``analysis/paper/out/runs_index.csv``       (S1, the matched-runs index)
  * ``analysis/paper/out/bootstrap.csv``        (S2, the paired bootstrap)
  * ``analysis/paper/out/time_calibration.csv`` (S1b, the legacy->matched factors)
  * raw saved metadata under ``runpod_results/`` (legacy ``config.json`` /
    ``phase_metrics.csv`` / ``timings.csv`` and the Y1 test predictions).

Nothing here trains, fits, evaluates or selects.  The script asserts ``torch``
and ``transformers`` are absent from ``sys.modules`` at import and at exit, and
never imports ``bio_is_curriculum`` (nor ``.selection`` / ``.curriculum`` /
``.models``).

Outputs (analysis/paper/out/):
  tables/tab_22_cell_means.tex      Table 2  (``\\label{tab:22-cell-means}``)
  tables/tab_cost_breakdown.tex     cost per stage + tokens/visits
  tables/tab_frozen_selection.tex   Y3 frozen-selection schedules
  tables/tab_per_class.tex          per-class Books / Reuters-90 groups
  tables/tab_beta_theta.tex         Y5 beta x theta grid
  numbers.json                      every cell keyed for the TeX <-> numbers lint

Conventions
-----------
  * Every emitted table row ENDS with a stable trace comment ``% k:<key>``.
  * A legacy time cell that was rescaled carries ``cal:<key>`` in that comment;
    its uncalibrated value is kept under ``legacy_raw`` in ``numbers.json``.
  * Y1-Y5 times are NEVER rescaled (only legacy cells are).
  * Medline stays ``\\LEGACY[pending]``: its cells are pending-marked and its
    times are the legacy cells rescaled here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# --- hard no-torch / no-transformers guard (checked again at exit) -----------
for _mod in ("torch", "transformers"):
    assert _mod not in sys.modules, f"{_mod} must not be imported by this script"

import numpy as np
import pandas as pd

# The NI cell of Table 2 is decided by the pre-committed gate rule, which lives
# with the gates (``gates.ni_label``); sharing it keeps make_tables.py output and
# ``out/gates.csv`` in agreement (asserted by lint_tex.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gates import ni_label  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants (dataset order V8; plan section 2)
# --------------------------------------------------------------------------- #

DATASET_ORDER = ["yelp_2013", "agnews", "dblp", "books", "reuters90"]
DATASET_LABEL = {
    "yelp_2013": "Yelp-2013", "agnews": "AG News", "dblp": "DBLP",
    "books": "Books", "reuters90": "Reuters-90", "medline": "Medline",
}
Y1_ARMS = ["raw", "is", "cl", "is_cl"]
Y1_ARM_LABEL = {"raw": "Raw", "is": "IS", "cl": "CL", "is_cl": "SPURS"}
DELTA_COMPARISON = {"is": "IS-Raw", "cl": "CL-Raw", "is_cl": "SPURS-Raw"}

Y3_ARM_ORDER = ["sel_composite", "sel_margin_only", "sel_entropy_only", "sel_random_phases"]
Y3_ARM_LABEL = {
    "sel_composite": "Composite", "sel_margin_only": "Margin only",
    "sel_entropy_only": "Entropy only", "sel_random_phases": "Random",
}
Y3_DELTA_COMPARISON = {
    "sel_margin_only": "MarginOnly", "sel_entropy_only": "EntropyOnly",
    "sel_random_phases": "RandomPhases",
}

BETAS = ["0.1", "0.3", "0.5"]
THETAS = ["0.0", "0.2", "0.4"]
Y5_DATASETS = ["books", "reuters90"]

COST_DATASETS = ["reuters90", "yelp_2013"]      # body: smallest + largest
PERCLASS_DATASETS = ["books", "reuters90"]
MEDLINE_MARK = "\\LEGACY[pending]"
MEDLINE_FACTORIAL_DIR = "medline-5cv-20260920-172720"
# Medline Raw: the single clean roberta-base bs-16 run whose fold mean matches
# the skeleton (0.822); a second such run exists but is not the reference.
MEDLINE_RAW_GLOB = "medline-5cv-20260828-210758/raw_fold*/phase_metrics.csv"
MEDLINE_SPURS_DELTA_LEGACY = -0.006  # legacy fallback printed in the skeleton
# The Medline skeleton values (Raw 36,349 / IS 34,270 / CL 27,058 / SPURS 19,790)
# are the *legacy* medians.  IS/CL/SPURS reproduce medians of the synced shard
# ``medline-5cv-20260920-172720`` exactly when the re-downloaded ``*_with_pred``
# folds are removed (the 5-fold dirs mix two fold-0 copies); the Raw cell came
# from a different, roberta-base legacy run that is not in this snapshot, so its
# legacy total is taken from the skeleton and only its *time* is rescaled.
MEDLINE_LEGACY_TOTAL_S = {"raw": 36349.0, "is": 34270.0, "cl": 27058.0, "is_cl": 19790.0}

T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def f3(x: float) -> str:
    return "0.000" if x == 0.0 else f"{x:.3f}"


def f_secs(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")


def f_f1_pm(f1: float, half: float) -> str:
    return f"${f3(f1)} \\pm {f3(half)}$"


def f_ret(x: float) -> str:
    return f"{x:.2f}"


def f_m(x: float) -> str:
    return f"{x / 1e6:.2f}M"


def f_delta_ci(d: float, lo: float, hi: float) -> str:
    return f"${d:.3f}$ [${lo:.3f}$, ${hi:.3f}$]"


def sort_dataset_rows(datasets) -> list[str]:
    rank = {d: i for i, d in enumerate(DATASET_ORDER)}
    return sorted(datasets, key=lambda d: (rank.get(d, 99), d))


def _sanitize(obj):
    """Recursively replace non-finite floats with ``None`` (valid JSON)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def t_legacy_halfwidth(vals) -> float:
    """95% t half-width of a fold list (median-based for the paper's decimal 2dp)."""
    a = np.asarray(vals, dtype=float)
    n = len(a)
    if n < 2:
        return 0.0
    se = float(a.std(ddof=1)) / math.sqrt(n)
    return T_CRIT_95.get(n - 1, 1.96) * se


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


class Tables:
    def __init__(self, index, boot, calibration, results_dir):
        self.index = index
        self.boot = boot
        self.calibration = calibration
        self.results_dir = Path(results_dir)
        self.rows: list[dict] = []
        self.tex: dict[str, list[str]] = {}
        fac = calibration[calibration.record_type == "factor"]
        probe = calibration[calibration.record_type == "probe_factor"]
        self.c_rest = float(fac[fac.key == "c_rest"].value.iloc[0])
        self.total_ratio = float(fac[fac.key == "total_ratio_geomean"].value.iloc[0])
        self.c_probe = {str(r.dataset): float(r.value)
                        for r in probe.itertuples(index=False)}
        self.meta = {
            "protocol": {
                "ci_level": float(boot.ci_level.iloc[0]),
                "delta": float(boot.delta.iloc[0]),
                "n_bootstrap": int(boot.n_bootstrap.iloc[0]),
                "seed": int(boot.seed.iloc[0]),
            },
            "dataset_order": DATASET_ORDER,
            "calibration": {"c_rest": self.c_rest, "total_ratio": self.total_ratio,
                            "c_probe": self.c_probe},
            "provenance": "aggregations of runs_index.csv / bootstrap.csv / "
                          "time_calibration.csv + raw saved metadata",
        }

    # -- emit rows / keys ----------------------------------------------------

    def add_row(self, table: str, body: str, key: str, cells: dict,
                prefix: str = "", cal_key: str | None = None, legacy: bool = False,
                status: str = "matched", legacy_raw: dict | None = None) -> None:
        trace = f"  % k:{key}" + (f" cal:{cal_key}" if cal_key else "")
        head = f"{prefix}\n      " if prefix else "  "
        self.tex.setdefault(table, []).append(f"{head}{body} \\\\{trace}")
        self.rows.append(self._rec(key, table, cells, status, cal_key, legacy,
                                    legacy_raw))

    def add_key(self, table: str, key: str, cells: dict, status: str = "matched") -> None:
        self.rows.append(self._rec(key, table, cells, status, None, False, None))

    @staticmethod
    def _rec(key, table, cells, status, cal_key, legacy, legacy_raw) -> dict:
        rec = {"key": key, "table": table, "status": status,
               "cells": _sanitize(cells), "calibrated_time_cell": bool(cal_key),
               "legacy": legacy}
        if cal_key:
            rec["legacy_raw"] = _sanitize(legacy_raw or {})
        return rec

    # -- primitives ----------------------------------------------------------

    def _cell(self, campaign, dataset, arm, column) -> pd.Series:
        return self.index[(self.index.campaign == campaign)
                          & (self.index.dataset == dataset)
                          & (self.index.arm == arm)][column]

    def fmean(self, campaign, dataset, arm, column) -> float:
        s = self._cell(campaign, dataset, arm, column)
        return float(s.mean()) if len(s) else float("nan")

    def t_halfwidth(self, campaign, dataset, arm, column="f1_last_phase") -> float:
        s = self._cell(campaign, dataset, arm, column).astype(float)
        n = len(s)
        if n < 2:
            return float("nan")
        se = float(s.std(ddof=1)) / math.sqrt(n)
        return T_CRIT_95.get(n - 1, 1.96) * se

    def boot_row(self, comparison) -> dict:
        m = self.boot[self.boot.comparison == comparison]
        if len(m) != 1:
            raise SystemExit(f"BLOCKED: comparison {comparison!r} -> {len(m)} rows")
        return m.iloc[0].to_dict()


# --------------------------------------------------------------------------- #
# Legacy shards (Medline)
# --------------------------------------------------------------------------- #


def read_legacy_shards(results_dir: Path, pattern: str,
                       require_clean_folds: bool = False) -> list[dict]:
    out = []
    seen_folds = set()
    for pm_path in sorted(results_dir.glob(pattern)):
        d = pm_path.parent
        fold = int(str(pm_path).split("fold")[-1].split("/")[0])
        # The synced 5-fold dirs contain a re-downloaded duplicate of fold 0, so a
        # pattern strict over folds rejects a directory that repeats a fold number
        # (the IS/CL/SPURS bases must be one shard per fold, 5 in total).
        if fold in seen_folds:
            if require_clean_folds:
                raise SystemExit(
                    f"BLOCKED: mixed/duplicate folds under {pattern} ({pm_path})")
            continue
        seen_folds.add(fold)
        pm = pd.read_csv(pm_path, encoding="utf-8-sig")
        timings: dict[str, float] = {}
        with open(d / "timings.csv", encoding="utf-8-sig", newline="") as fh:
            for row in csv.reader(fh):
                if len(row) >= 2 and row[0] != "name":
                    timings[row[0]] = float(row[1])
        out.append({"dir": str(d), "f1": float(pm["macro_f1"].iloc[-1]),
                    "total_s": timings.get("total_run_time_s", float("nan")),
                    "probe_s": timings.get("is_fit_time_s", float("nan"))})
    if not out:
        raise SystemExit(f"BLOCKED: no legacy shards for {pattern}")
    return out


# --------------------------------------------------------------------------- #
# Table 2 - matched 2^2 cell means
# --------------------------------------------------------------------------- #


def table_22(t: Tables) -> None:
    tb = "tab_22_cell_means"
    for ds in sort_dataset_rows(DATASET_ORDER):
        i_boot = t.boot_row(f"I/{ds}")
        i_cell = ("\\multirow{4}{*}{" + f_delta_ci(
            i_boot["delta_hat"], i_boot["ci_low"], i_boot["ci_high"]) + "}")
        t.add_key(tb, f"Y1/{ds}/I", {
            "interaction_I": i_boot["delta_hat"], "interaction_ci_low": i_boot["ci_low"],
            "interaction_ci_high": i_boot["ci_high"], "noise_floor": i_boot["noise_floor"]})
        for idx, arm in enumerate(Y1_ARMS):
            f1 = t.fmean("Y1", ds, arm, "f1_last_phase")
            half = t.t_halfwidth("Y1", ds, arm)
            total = t.fmean("Y1", ds, arm, "total_run_time_s")
            ret = 1.0 if arm in ("raw", "cl") else t.fmean(
                "Y1", ds, arm, "retention_effective")
            key = f"Y1/{ds}/{Y1_ARM_LABEL[arm]}"
            cells = {"f1_foldmean": f1, "f1_halfwidth": half, "total_s": total,
                     "retention": ret, "arm": Y1_ARM_LABEL[arm]}
            last = ""
            if arm == "raw":
                delta_cell, ni_cell, last = "--", "--", i_cell
                prefix = f"    \\multirow{{4}}{{*}}{{{DATASET_LABEL[ds]}}}" if idx == 0 else ""
            else:
                b = t.boot_row(f"{DELTA_COMPARISON[arm]}/{ds}")
                delta_cell = f_delta_ci(b["delta_hat"], b["ci_low"], b["ci_high"])
                cells.update(delta=b["delta_hat"], ci_low=b["ci_low"], ci_high=b["ci_high"])
                if arm == "is_cl":
                    ni_cell = ni_label(b["ci_low"], b["ci_high"], b["delta"])
                    cells.update(ni=bool(b["ni_pass"]),
                                 equivalence=bool(b["equivalence_pass"]),
                                 ni_status=b["equivalence_status"])
                    last = ""  # the interaction I multirow sits on the Raw row only
                else:
                    ni_cell = "--"
                prefix = ""
            body = (f"& {Y1_ARM_LABEL[arm]} & {f_ret(ret)} & {f_f1_pm(f1, half)} & "
                    f"{f_secs(total)} & {delta_cell} & {ni_cell} & {last}")
            t.add_row(tb, body, key, cells, prefix=prefix)

    # --- Medline: legacy pending, legacy times rescaled ---------------------
    ds = "medline"
    cp = t.c_probe["medline"]
    # one shard per fold (5 total); a repeated fold number would be a mixed dir
    legacy = {
        a: read_legacy_shards(
            t.results_dir, f"{MEDLINE_FACTORIAL_DIR}/{a}_fold*/phase_metrics.csv",
            require_clean_folds=True)
        for a in ("is", "cl", "is_cl")}
    legacy["raw"] = read_legacy_shards(t.results_dir, MEDLINE_RAW_GLOB)
    for idx, arm in enumerate(Y1_ARMS):
        shards = legacy[arm]
        f1 = float(np.mean([s["f1"] for s in shards]))               # skeleton uses mean ± half-width
        half = t_legacy_halfwidth([s["f1"] for s in shards])
        total = MEDLINE_LEGACY_TOTAL_S[arm]          # skeleton legacy total
        probe = float(np.median([s["probe_s"] for s in shards]))
        t_cal = total / t.c_rest if arm == "raw" else probe / cp + (total - probe) / t.c_rest
        key = f"legacy/medline/{Y1_ARM_LABEL[arm]}"
        f1_cell = f"{MEDLINE_MARK}{{${f3(f1)} \\pm {f3(half)}$}}"
        time_cell = f"{MEDLINE_MARK}{{{f_secs(t_cal)}}}"
        ret_cell = "1.00" if arm in ("raw", "cl") else "--"
        if arm == "is_cl":
            delta_cell = f"{MEDLINE_MARK}{{${MEDLINE_SPURS_DELTA_LEGACY:.3f}$}}"
        else:
            delta_cell = "--"
        last = "\\multirow{4}{*}{--}" if arm == "raw" else ""
        prefix = (f"    \\multirow{{4}}{{*}}{{{DATASET_LABEL[ds]}}}" if idx == 0 else "")
        body = (f"& {Y1_ARM_LABEL[arm]}   & {ret_cell} & {f1_cell} & {time_cell} & "
                f"{delta_cell} & -- & {last}")
        cells = {"f1_legacy": f1, "f1_halfwidth_legacy": half,
                 "probe_s_legacy": None if math.isnan(probe) else probe,
                 "total_s_legacy": total, "total_s_calibrated": t_cal, "pending": True,
                 "source_dir": shards[0]["dir"],
                 "encoder_note": "legacy roberta-base Raw; ModernBERT IS/CL/SPURS"}
        t.add_row(tb, body, key, cells, prefix=prefix, cal_key=key, legacy=True,
                  status="pending",
                  legacy_raw={"total_s": total,
                              "probe_s": None if math.isnan(probe) else probe})


# --------------------------------------------------------------------------- #
# Cost breakdown table
# --------------------------------------------------------------------------- #


def table_cost(t: Tables) -> None:
    tb = "tab_cost_breakdown"
    for ds in COST_DATASETS:
        for idx, arm in enumerate(Y1_ARMS):
            key = f"Y1/{ds}/{Y1_ARM_LABEL[arm]}/cost"
            load = t.fmean("Y1", ds, arm, "cost_load")
            probe = t.fmean("Y1", ds, arm, "cost_probe_sel")
            fine = t.fmean("Y1", ds, arm, "cost_finetune")
            ev = t.fmean("Y1", ds, arm, "cost_eval")
            total = t.fmean("Y1", ds, arm, "total_run_time_s")
            tok = t.fmean("Y1", ds, arm, "tokens")
            vis = t.fmean("Y1", ds, arm, "visits")
            assert ev >= -1e-6, f"negative evaluation cost {key}: {ev}"
            assert abs(load + probe + fine + max(ev, 0.0) - total) < 1e-4, key
            cells = {"load_s": load, "probe_selection_s": probe, "finetune_s": fine,
                     "eval_s": max(ev, 0.0), "total_s": total, "tokens": tok,
                     "visits": vis, "tokens_M": tok / 1e6, "visits_M": vis / 1e6,
                     "arm": Y1_ARM_LABEL[arm]}
            prefix = (f"    \\multirow{{4}}{{*}}{{{DATASET_LABEL[ds]}}}" if idx == 0 else "")
            body = (f"& {Y1_ARM_LABEL[arm]}   & {f_secs(load)} & {f_secs(probe)}  & "
                    f"{f_secs(fine)} & {f_secs(max(ev, 0.0))} & {f_secs(total)} & "
                    f"{f_m(tok)} & {f_m(vis)}")
            t.add_row(tb, body, key, cells, prefix=prefix)


# --------------------------------------------------------------------------- #
# Frozen-selection table (Y3)
# --------------------------------------------------------------------------- #


def table_frozen(t: Tables) -> None:
    tb = "tab_frozen_selection"
    for ds in sort_dataset_rows(["agnews", "books", "reuters90"]):
        for idx, arm in enumerate(Y3_ARM_ORDER):
            f1 = t.fmean("Y3", ds, arm, "f1_last_phase")
            half = t.t_halfwidth("Y3", ds, arm)
            hard = t.fmean("Y3", ds, arm, "hard_f1")
            key = f"Y3/{ds}/{Y3_ARM_LABEL[arm]}"
            cells = {"f1_foldmean": f1, "f1_halfwidth": half, "hard_f1": hard,
                     "arm": Y3_ARM_LABEL[arm]}
            if arm == "sel_composite":
                delta_cell = "--"
            else:
                b = t.boot_row(f"{Y3_DELTA_COMPARISON[arm]}-Composite/{ds}")
                delta_cell = f_delta_ci(b["delta_hat"], b["ci_low"], b["ci_high"])
                cells.update(delta=b["delta_hat"], ci_low=b["ci_low"], ci_high=b["ci_high"],
                             noise_floor=b["noise_floor"], below_noise=bool(b["below_noise"]))
            prefix = (f"    \\multirow{{4}}{{*}}{{{DATASET_LABEL[ds]}}}" if idx == 0 else "")
            body = f"& {Y3_ARM_LABEL[arm]:<12} & {f_f1_pm(f1, half)} & {f3(hard)} & {delta_cell}"
            t.add_row(tb, body, key, cells, prefix=prefix)


# --------------------------------------------------------------------------- #
# Per-class table (Y1, Books / Reuters-90)
# --------------------------------------------------------------------------- #


def group_membership(support: dict[str, int], n_parts: int = 3) -> dict[str, int]:
    items = sorted(support.items(), key=lambda kv: (int(kv[1]), int(kv[0])))
    n = len(items)
    gof = {}
    for g in range(n_parts):
        for kk, _ in items[(g * n) // n_parts:((g + 1) * n) // n_parts]:
            gof[kk] = g
    return gof


def perclass_f1(results_dir: Path, index, ds, arm, labels) -> dict[int, float]:
    lab_idx = {int(c): i for i, c in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=np.int64)
    sub = index[(index.campaign == "Y1") & (index.dataset == ds) & (index.arm == arm)]
    for d in sub["dir"]:
        p = results_dir.parent / d if not Path(d).is_absolute() else Path(d)
        pred = pd.read_csv(p / "predictions_test.csv", encoding="utf-8-sig")
        yt = np.array([lab_idx[int(v)] for v in pred["y_true"]], dtype=np.int64)
        yp = np.array([lab_idx[int(v)] for v in pred["y_pred"]], dtype=np.int64)
        cm += np.bincount(yt * k + yp, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return {int(labels[i]): float(f1[i]) for i in range(k)}


def table_per_class(t: Tables) -> None:
    tb = "tab_per_class"
    for ds in PERCLASS_DATASETS:
        sub = t.index[(t.index.campaign == "Y1") & (t.index.dataset == ds)
                      & (t.index.arm == "is")]
        support: dict[str, int] = {}
        removed: dict[str, int] = {}
        for _, r in sub.iterrows():
            tot = json.loads(r["total_by_class"]) if isinstance(r["total_by_class"], str) else {}
            rem = json.loads(r["removed_by_class"]) if isinstance(r["removed_by_class"], str) else {}
            for kk, v in tot.items():
                support[kk] = support.get(kk, 0) + int(v)
            for kk, v in rem.items():
                removed[kk] = removed.get(kk, 0) + int(v)
        gof = group_membership(support, 3)
        labels = sorted(support, key=lambda c: int(c))
        arms_f1 = {arm: perclass_f1(t.results_dir, t.index, ds, arm, labels)
                   for arm in ("raw", "is", "is_cl")}
        for idx, (grp, gid) in enumerate((("Rare", 0), ("Frequent", 2))):
            cls = [c for c in labels if gof[c] == gid]
            g_removed = sum(removed.get(c, 0) for c in cls)
            g_total = sum(support.get(c, 0) for c in cls)
            raw_f1 = float(np.mean([arms_f1["raw"][int(c)] for c in cls]))
            is_f1 = float(np.mean([arms_f1["is"][int(c)] for c in cls]))
            spurs_f1 = float(np.mean([arms_f1["is_cl"][int(c)] for c in cls]))
            pct = 100.0 * g_removed / g_total if g_total else float("nan")
            key = f"Y1/{ds}/{grp}"
            cells = {"raw_f1_pooled_perclass": raw_f1, "is_f1_pooled_perclass": is_f1,
                     "spurs_f1_pooled_perclass": spurs_f1, "pct_removed": pct,
                     "n_classes_group": len(cls), "group_support": g_total,
                     "group_removed": g_removed,
                     "definition": "bottom/top tercile of classes by training support "
                                   "(total_by_class); cell = mean per-class F1 over "
                                   "pooled predictions"}
            prefix = (f"    \\multirow{{2}}{{*}}{{{DATASET_LABEL[ds]}}}" if idx == 0 else "")
            body = (f"& {grp:<8} & {f3(raw_f1)} & {f3(is_f1)} & {f3(spurs_f1)} & {pct:.1f}")
            t.add_row(tb, body, key, cells, prefix=prefix)


# --------------------------------------------------------------------------- #
# beta x theta table (Y5)
# --------------------------------------------------------------------------- #


def table_beta_theta(t: Tables) -> None:
    tb = "tab_beta_theta"
    for ds in Y5_DATASETS:
        for idx, beta in enumerate(BETAS):
            vals, cells = [], {"dataset": ds, "beta": beta}
            for theta in THETAS:
                arm = f"sweep_b{beta}_t{theta}"
                f1 = t.fmean("Y5", ds, arm, "f1_last_phase")
                ret = t.fmean("Y5", ds, arm, "retention_effective")
                cells[f"theta{theta}"] = {"f1": f1, "retention": ret}
                val = f"{f3(f1)} ({ret:.2f})"
                if (beta, theta) == ("0.3", "0.2"):
                    val = "\\underline{" + val + "}"
                vals.append(val)
            key = f"Y5/{ds}/beta{beta}"
            prefix = (f"    \\multirow{{3}}{{*}}{{{DATASET_LABEL[ds]}}}" if idx == 0 else "")
            body = f"& $\\beta{{=}}{beta}$ & " + " & ".join(vals)
            t.add_row(tb, body, key, cells, prefix=prefix)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description="S4 render the table skeletons.")
    ap.add_argument("--index", default="analysis/paper/out/runs_index.csv")
    ap.add_argument("--bootstrap", default="analysis/paper/out/bootstrap.csv")
    ap.add_argument("--calibration", default="analysis/paper/out/time_calibration.csv")
    ap.add_argument("--results-dir", default="runpod_results")
    ap.add_argument("--out-dir", default="analysis/paper/out")
    args = ap.parse_args()

    index = pd.read_csv(args.index)
    boot = pd.read_csv(args.bootstrap)
    calibration = pd.read_csv(args.calibration)
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    t = Tables(index, boot, calibration, args.results_dir)
    table_22(t)
    table_cost(t)
    table_frozen(t)
    table_per_class(t)
    table_beta_theta(t)

    # --- validate before writing -------------------------------------------
    tex_keys = [ln.split("% k:")[1].split()[0].split("cal:")[0].strip()
                for ln in (l for ls in t.tex.values() for l in ls)
                if "\\" in ln and "% k:" in ln]
    dupes = sorted({k for k in tex_keys if tex_keys.count(k) > 1})
    assert not dupes, f"BLOCKED: duplicate trace keys: {dupes}"
    assert t.rows, "BLOCKED: no rows emitted"
    rank = {d: i for i, d in enumerate(DATASET_ORDER)}
    y1ranks = [rank[r["key"].split("/")[1]] for r in t.rows
               if r["key"].startswith("Y1/") and r["table"] == "tab_22_cell_means"]
    assert y1ranks == sorted(y1ranks), "BLOCKED: Table-2 dataset order != V8"
    for ln in (l for ls in t.tex.values() for l in ls):
        if ln.strip().startswith("%"):
            continue
        assert "% k:" in ln, f"row without % k: trace: {ln}"
        if "cal:" in ln:
            assert MEDLINE_MARK in ln, f"cal: outside a legacy cell: {ln}"

    written = []
    for name, lines in t.tex.items():
        path = tables_dir / f"{name}.tex"
        path.write_text(
            f"% S4 auto-generated rows for {name} (analysis/paper/make_tables.py)\n"
            + "\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)

    numbers_path = out_dir / "numbers.json"
    numbers_path.write_text(
        json.dumps({"meta": t.meta, "rows": t.rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    for mod in ("torch", "transformers"):
        assert mod not in sys.modules, f"{mod} leaked into sys.modules"

    print(f"wrote {len(written)} table fragments to {tables_dir}")
    for p in written:
        print(f"  {p}")
    n_cal = sum(1 for r in t.rows if r["calibrated_time_cell"])
    n_legacy = sum(1 for r in t.rows if r["legacy"])
    print(f"wrote {numbers_path} ({len(t.rows)} keyed rows); "
          f"calibrated legacy cells={n_cal}, legacy rows={n_legacy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
