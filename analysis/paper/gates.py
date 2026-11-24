#!/usr/bin/env python
"""S6 - apply the pre-committed gates mechanically and emit ``out/gates.csv``.

Read-only post-processing of the validated S2 bootstrap (``bootstrap.csv``) and,
for the cost condition of G-F, the S1 index (``runs_index.csv``).  The verdicts
are computed from the plan's gate table (paper/acl_latex_new_experiments_plan.md
section 2); no prose claim may be written before this file exists.

  G-A  SPURS - Raw, per dataset (Y1)        PASS if CI_low >= -delta
  G-G  interaction I (Y1)                    PASS if the CI excludes 0
  G-E  IS - Random@ret (Y1 x Y4)             PASS if CI_low > 0; CI subset of
                                             +/-delta -> "selection ~ random";
                                             otherwise INCONCLUSIVE
  G-F  SPURS - Raw@2, SPURS - Raw@4 + cost   PASS if CI_low > 0 for both at <= cost;
                                             Raw@2 within +/-delta at <= cost ->
                                             fallback (FAIL); else INCONCLUSIVE
  G-C  Random phases - Composite (Y3)        PASS if CI_high < 0; CI subset of
                                             +/-delta -> "order does not matter"; else
                                             INCONCLUSIVE
  G-D  Margin - Comp. AND Entropy - Comp.    PASS if both CI_high < 0; a term ties
      (Y3)                                   (CI subset of +/-delta) -> FAIL; else
                                             INCONCLUSIVE
  G-H  Y5 theta>0 vs theta=0 at each beta    PASS if some theta>0 delta > noise floor
      (Books primary, Reuters-90 secondary)  (descriptive, 3 folds); else FAIL

Every row carries a ``source_comparison`` (the bootstrap comparison(s) it reads)
and a verdict in {PASS, FAIL, INCONCLUSIVE}.  Asserts torch/transformers are
absent from ``sys.modules`` and never imports ``bio_is_curriculum``.

Output: analysis/paper/out/gates.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# --- hard no-torch / no-transformers guard (checked again at exit) -----------
for _mod in ("torch", "transformers"):
    assert _mod not in sys.modules, f"{_mod} must not be imported by this script"

import numpy as np
import pandas as pd

VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE"}
ALL_DATASETS = ["yelp_2013", "agnews", "dblp", "books", "reuters90"]
Y2_DATASETS = ["agnews", "books", "reuters90"]
Y5_DATASETS = ["books", "reuters90"]  # Books primary, Reuters-90 secondary
BETAS = ["0.1", "0.3", "0.5"]
THETAS = ["0.2", "0.4"]


class Boot:
    def __init__(self, boot: pd.DataFrame, index: pd.DataFrame):
        self.boot = boot.set_index("comparison")
        self.index = index
        self.delta = float(boot["delta"].iloc[0])
        self.ci_level = float(boot["ci_level"].iloc[0])

    def row(self, comparison: str) -> pd.Series:
        if comparison not in self.boot.index:
            raise SystemExit(f"BLOCKED: bootstrap comparison missing: {comparison}")
        return self.boot.loc[comparison]

    def noise_floor(self, dataset: str) -> float:
        s = self.boot[self.boot["dataset"] == dataset]["noise_floor"]
        return float(s.iloc[0]) if len(s) and pd.notna(s.iloc[0]) else float("nan")

    def total_s(self, campaign: str, dataset: str, arm: str) -> float:
        s = self.index[(self.index.campaign == campaign)
                       & (self.index.dataset == dataset)
                       & (self.index.arm == arm)]["total_run_time_s"]
        return float(s.mean()) if len(s) else float("nan")


def _within(lo: float, hi: float, delta: float) -> bool:
    """CI subset of [-delta, +delta] (the equivalence window)."""
    return lo >= -delta and hi <= delta


def verdict_ni(r) -> str:
    return "PASS" if r["ci_low"] >= -r["delta"] else "FAIL"


def verdict_positive_excludes_zero(r) -> str:
    """Synergy needs a POSITIVE interaction whose CI excludes 0 (plan: shorter
    time alone is not synergy, so a significantly negative I is a FAIL)."""
    return "PASS" if r["ci_low"] > 0.0 else "FAIL"


def verdict_positive(r) -> str:
    if r["ci_low"] > 0.0:
        return "PASS"
    if _within(r["ci_low"], r["ci_high"], r["delta"]):
        return "FAIL"
    return "INCONCLUSIVE"


def verdict_negative(r) -> str:
    if r["ci_high"] < 0.0:
        return "PASS"
    if _within(r["ci_low"], r["ci_high"], r["delta"]):
        return "FAIL"
    return "INCONCLUSIVE"


def build_rows(b: Boot) -> list[dict]:
    rows: list[dict] = []

    def add(gate, dataset, verdict, criterion, source, statistic, lo, hi,
            noise=None, cost_ok=None, detail="") -> None:
        assert verdict in VERDICTS, f"bad verdict {verdict!r}"
        assert source, f"gate {gate} has no source comparison"
        rows.append({
            "gate": gate, "dataset": dataset, "verdict": verdict,
            "criterion": criterion, "source_comparison": source,
            "statistic_delta": statistic, "ci_low": lo, "ci_high": hi,
            "delta": b.delta, "ci_level": b.ci_level,
            "noise_floor": noise, "cost_ok": cost_ok, "detail": detail,
            "n_instances": int(b.row(source.split(";")[0])["n_instances"]),
            "n_folds": int(b.row(source.split(";")[0])["n_folds"]),
        })

    # --- G-A: SPURS - Raw, per dataset (NI: CI_low >= -delta) ---------------
    for ds in ALL_DATASETS:
        cmp = f"SPURS-Raw/{ds}"
        r = b.row(cmp)
        v = verdict_ni(r)
        det = "NI holds" if v == "PASS" else (
            "NI not established" if r["ci_high"] >= -b.delta else "NI fails (No)")
        add("G-A", ds, v, "CI_low >= -delta", cmp, r["delta_hat"],
            r["ci_low"], r["ci_high"], noise=b.noise_floor(ds), detail=det)

    # --- G-G: interaction I excludes 0 -------------------------------------
    for ds in ALL_DATASETS:
        cmp = f"I/{ds}"
        r = b.row(cmp)
        v = verdict_positive_excludes_zero(r)
        add("G-G", ds, v, "interaction CI excludes 0 (positive)", cmp, r["delta_hat"],
            r["ci_low"], r["ci_high"], noise=b.noise_floor(ds),
            detail="synergy" if v == "PASS" else "no synergy")

    # --- G-E: IS - Random@ret > 0 ------------------------------------------
    for ds in Y2_DATASETS:
        cmp = f"IS-Random@ret/{ds}"
        r = b.row(cmp)
        v = verdict_positive(r)
        det = {"PASS": "selection > random at equal retention",
               "FAIL": "selection ~ random at equal retention",
               "INCONCLUSIVE": "not established"}[v]
        add("G-E", ds, v, "CI_low > 0", cmp, r["delta_hat"], r["ci_low"],
            r["ci_high"], noise=b.noise_floor(ds), detail=det)

    # --- G-F: SPURS - Raw@2 / Raw@4 at <= cost -----------------------------
    for ds in Y2_DATASETS:
        c2, c4 = f"SPURS-Raw@2/{ds}", f"SPURS-Raw@4/{ds}"
        r2, r4 = b.row(c2), b.row(c4)
        spurs_t = b.total_s("Y1", ds, "is_cl")
        raw2_t = b.total_s("Y2", ds, "raw_ep2")
        raw4_t = b.total_s("Y2", ds, "raw_ep4")
        cost_ok = bool(spurs_t <= raw2_t and spurs_t <= raw4_t)
        both_pos = r2["ci_low"] > 0.0 and r4["ci_low"] > 0.0
        if both_pos and cost_ok:
            v = "PASS"
        elif cost_ok and _within(r2["ci_low"], r2["ci_high"], b.delta):
            v = "FAIL"  # Raw@2 within +/-delta at <= cost -> fallback
        elif r2["ci_high"] < 0.0:
            v = "FAIL"  # Raw@2 significantly better at <= cost: stronger fallback
        else:
            v = "INCONCLUSIVE"
        stat = float(min(r2["delta_hat"], r4["delta_hat"]))
        add("G-F", ds, v, "CI_low > 0 for both AND cost <= control",
            f"{c2};{c4}", stat, float(min(r2["ci_low"], r4["ci_low"])),
            float(min(r2["ci_high"], r4["ci_high"])), noise=b.noise_floor(ds),
            cost_ok=cost_ok,
            detail=f"spurs={spurs_t:.0f}s raw@2={raw2_t:.0f}s raw@4={raw4_t:.0f}s")

    # --- G-C: random phases lose to composite ------------------------------
    for ds in Y2_DATASETS:
        cmp = f"RandomPhases-Composite/{ds}"
        r = b.row(cmp)
        v = verdict_negative(r)
        det = {"PASS": "order matters", "FAIL": "order does not matter at delta",
               "INCONCLUSIVE": "not established"}[v]
        add("G-C", ds, v, "CI_high < 0", cmp, r["delta_hat"], r["ci_low"],
            r["ci_high"], noise=b.noise_floor(ds), detail=det)

    # --- G-D: composite adds over both single terms ------------------------
    for ds in Y2_DATASETS:
        cm, ce = f"MarginOnly-Composite/{ds}", f"EntropyOnly-Composite/{ds}"
        rm, re_ = b.row(cm), b.row(ce)
        if rm["ci_high"] < 0.0 and re_["ci_high"] < 0.0:
            v = "PASS"
        elif _within(rm["ci_low"], rm["ci_high"], b.delta) or \
                _within(re_["ci_low"], re_["ci_high"], b.delta):
            v = "FAIL"  # a single term ties -> composite not shown to add
        else:
            v = "INCONCLUSIVE"
        stat = float(max(rm["delta_hat"], re_["delta_hat"]))
        add("G-D", ds, v, "both CI_high < 0", f"{cm};{ce}", stat,
            float(max(rm["ci_low"], re_["ci_low"])),
            float(max(rm["ci_high"], re_["ci_high"])), noise=b.noise_floor(ds),
            detail=f"margin_high={rm['ci_high']:.4f} entropy_high={re_['ci_high']:.4f}")

    # --- G-H: theta>0 helps (descriptive, delta > noise floor) -------------
    for ds in Y5_DATASETS:
        floor = b.noise_floor(ds)
        cmps, deltas = [], []
        for beta in BETAS:
            for theta in THETAS:
                cmp = f"theta{theta}-theta0.0/beta{beta}/{ds}"
                r = b.row(cmp)
                cmps.append(cmp)
                deltas.append(float(r["delta_hat"]))
        v = "PASS" if any(d > floor for d in deltas) else "FAIL"
        stat = float(max(deltas))
        add("G-H", ds, v, "a theta>0 delta > noise floor",
            ";".join(cmps), stat, float(min(
                b.row(c)["ci_low"] for c in cmps)), float(max(
                b.row(c)["ci_high"] for c in cmps)), noise=floor,
            detail="theta not shown to help" if v == "FAIL" else "theta helps on some cell")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="S6 apply the pre-committed gates.")
    ap.add_argument("--bootstrap", default="analysis/paper/out/bootstrap.csv")
    ap.add_argument("--index", default="analysis/paper/out/runs_index.csv")
    ap.add_argument("--out", default="analysis/paper/out/gates.csv")
    args = ap.parse_args()

    boot = pd.read_csv(args.bootstrap)
    index = pd.read_csv(args.index)
    b = Boot(boot, index)
    rows = build_rows(b)
    df = pd.DataFrame(rows)

    # --- validation: each gate row has a verdict + a source comparison ------
    assert len(df) > 0
    assert set(df["verdict"]) <= VERDICTS, f"bad verdicts: {set(df['verdict'])}"
    assert df["source_comparison"].astype(str).str.len().gt(0).all()
    for _, r in df.iterrows():
        assert r["verdict"] in VERDICTS, r["gate"]
        for cmp in str(r["source_comparison"]).split(";"):
            assert cmp in b.boot.index, f"{r['gate']}/{r['dataset']} -> {cmp}"
    gates_present = set(df["gate"])
    assert gates_present == {"G-A", "G-G", "G-E", "G-F", "G-C", "G-D", "G-H"}, gates_present

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6g")

    for mod in ("torch", "transformers"):
        assert mod not in sys.modules, f"{mod} leaked into sys.modules"

    print(f"wrote {out} ({len(df)} rows)")
    for gate, g in df.groupby("gate", sort=False):
        counts = g["verdict"].value_counts().to_dict()
        print(f"[INFO] {gate}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
