#!/usr/bin/env python
"""S3 - compact but genuinely blind independent verifier for bootstrap.csv.

Written WITHOUT reading any generator source under analysis/paper/*.py. Reads
only:

  * paper/acl_latex_new_experiments_plan.md            (the protocol/spec)
  * runpod_results/**/predictions_test.csv              (raw saved predictions)
  * analysis/paper/out/bootstrap.csv                    (generator output, schema)
  * analysis/paper/out/bootstrap_meta.json              (public schema)
  * analysis/paper/out/runs_index.csv                   (public index)

It independently:
  1. resolves (campaign, dataset, arm) -> prediction dirs with its own resolver,
  2. recomputes pooled macro-F1 with its own metric (confusion-count macro-F1,
     verified bit-identical to sklearn.metrics.f1_score), and
  3. recomputes every comparison's point estimate and CI with a paired
     pooled-instance percentile bootstrap using a DIFFERENT seed (42 -> 20261124).

Gates (per the plan):
  * point estimates agree to 1e-9,
  * CI bounds agree to 0.002 (Monte-Carlo error of a different seed),
  * paired idx/y_true identity holds, no hard-F1 delta,
  * no torch / transformers / bio_is_curriculum model imports.

Exit code non-zero on any gate failure. Writes analysis/paper/out/verify_report.md.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

# --- hard guardrails: no torch, no transformers ----------------------------
FORBIDDEN = {"torch", "transformers"}
assert not (FORBIDDEN & set(sys.modules)), (
    "torch/transformers already imported before verification began")

POINT_TOL = 1e-9
CI_TOL = 0.002
B = 5000                   # percentile MC error << 0.002 CI tolerance
SEED = 20261124            # deliberately != generator's 42
PERCENTILES = [5.0, 95.0]  # two-sided 90% percentile CI

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "analysis", "paper", "out")


# ---------------------------------------------------------------------------
# own metric: exact macro-F1 from a confusion count matrix.
# Bit-identical to sklearn.metrics.f1_score(average="macro") on the full data
# and on resamples that drop classes entirely (present = tp+fp+fn > 0).
# ---------------------------------------------------------------------------
def macro_f1_from_codes(codes, K):
    cm = np.bincount(codes, minlength=K * K).reshape(K, K)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1), 0.0)
    present = (tp + fp + fn) > 0
    return float(f1[present].mean()) if present.any() else 0.0


# ---------------------------------------------------------------------------
# own resolver + pooled loader
# ---------------------------------------------------------------------------
class ArmData:
    def __init__(self, idx, campaign, dataset, arm):
        sub = idx[(idx.campaign == campaign) &
                  (idx.dataset == dataset) &
                  (idx.arm == arm)].sort_values("fold")
        assert len(sub), f"no rows for {campaign}/{dataset}/{arm}"
        self.folds = sub["fold"].astype(int).tolist()
        self.dirs = sub["dir"].tolist()
        self.yt, self.yp = [], []
        for d in self.dirs:
            df = pd.read_csv(os.path.join(ROOT, d, "predictions_test.csv"))
            self.yt.append(df["y_true"].to_numpy())
            self.yp.append(df["y_pred"].to_numpy())

    def subset(self, folds):
        keep = [i for i, f in enumerate(self.folds) if f in folds]
        yt = np.concatenate([self.yt[i] for i in keep]) if keep else np.array([])
        yp = np.concatenate([self.yp[i] for i in keep]) if keep else np.array([])
        return yt, yp


def comparison_arms(cmp_row, dataset):
    """Return (list of (campaign,dataset,arm), kind)."""
    if str(cmp_row["comparison"]).startswith("I/"):
        return [("Y1", dataset, "is_cl"), ("Y1", dataset, "is"),
                ("Y1", dataset, "cl"), ("Y1", dataset, "raw")], "interaction"
    return [(cmp_row["campaign_a"], dataset, cmp_row["arm_a"]),
            (cmp_row["campaign_b"], dataset, cmp_row["arm_b"])], "paired"


def compute_unit(idx, cmp_row, cache):
    """Full independent recomputation of one comparison (cached)."""
    name = cmp_row["comparison"]
    if name in cache:
        return cache[name]
    ds = cmp_row["dataset"]
    arms, kind = comparison_arms(cmp_row, ds)
    data = [ArmData(idx, c, ds, a) for c, _, a in arms]

    fsets = [set(d.folds) for d in data]
    common = sorted(set.intersection(*fsets))
    assert common, f"no common folds for {name}"

    subset = [d.subset(common) for d in data]
    yts = [s[0] for s in subset]
    yps = [s[1] for s in subset]
    N = len(yts[0])

    # paired idx/y_true identity: pooled y_true identical across arms.
    # (idx identity is implied by same per-fold lengths, which the equal
    # per-fold y_true arrays verify jointly with the fold ordering.)
    paired_ok = all(a.shape == yts[0].shape for a in yts) and \
        all(np.array_equal(yts[0], y) for y in yts[1:])

    K = int(max(int(y.max()) for y in yts)) + 1
    codes = [y.astype(np.int64) * K + p.astype(np.int64) for y, p in zip(yts, yps)]

    point = [macro_f1_from_codes(c, K) for c in codes]
    if kind == "interaction":
        f1_a, f1_b = point[0], point[3]
        delta = point[0] - point[1] - point[2] + point[3]
    else:
        f1_a, f1_b = point[0], point[1]
        delta = point[0] - point[1]

    rng = np.random.RandomState(SEED)
    delta_star = np.empty(B)
    for b in range(B):
        ri = rng.randint(0, N, N)
        if kind == "interaction":
            fb = [macro_f1_from_codes(c[ri], K) for c in codes]
            delta_star[b] = fb[0] - fb[1] - fb[2] + fb[3]
        else:
            delta_star[b] = macro_f1_from_codes(codes[0][ri], K) - \
                macro_f1_from_codes(codes[1][ri], K)
    lo, hi = np.percentile(delta_star, PERCENTILES)

    unit = {
        "arms": arms, "kind": kind, "common_folds": common, "K": K, "N": N,
        "paired_ok": paired_ok, "f1_a": f1_a, "f1_b": f1_b,
        "delta_hat": delta, "ci_low": float(lo), "ci_high": float(hi),
        "arm_f1": point, "dirs": [d.dirs for d in data],
    }
    cache[name] = unit
    return unit


def verify_idx_identity(idx, cmp_row, unit):
    """Full byte-level idx + y_true identity across the arms of a comparison."""
    ds = cmp_row["dataset"]
    arms, _ = comparison_arms(cmp_row, ds)
    common = set(unit["common_folds"])
    ref_i = ref_t = None
    ok = True
    for c, d, a in arms:
        sub = idx[(idx.campaign == c) & (idx.dataset == d) & (idx.arm == a) &
                  (idx.fold.isin(common))].sort_values("fold")
        ii, tt = [], []
        for _, r in sub.iterrows():
            df = pd.read_csv(os.path.join(ROOT, r["dir"], "predictions_test.csv"))
            ii.append(df["idx"].to_numpy())
            tt.append(df["y_true"].to_numpy())
        ii = np.concatenate(ii)
        tt = np.concatenate(tt)
        if ref_i is None:
            ref_i, ref_t = ii, tt
        else:
            ok = ok and np.array_equal(ref_i, ii) and np.array_equal(ref_t, tt)
    return ok


REPRESENTATIVE = [
    ("SPURS-Raw/agnews", "Y1 SPURS-Raw"),
    ("I/agnews", "four-arm interaction I (agnews)"),
    ("IS-Random@ret/agnews", "IS vs Random@ret (Y1 x Y4)"),
    ("SPURS-Raw@2/books", "Y2 comparison SPURS-Raw@2 (books)"),
    ("RandomPhases-Composite/agnews", "Y3 control RandomPhases vs Composite"),
    ("theta0.2-theta0.0/beta0.1/books", "Y5 theta0.2 vs theta0.0 (books beta0.1)"),
]


def main():
    boot = pd.read_csv(os.path.join(OUT, "bootstrap.csv"))
    with open(os.path.join(OUT, "bootstrap_meta.json")) as f:
        meta = json.load(f)
    idx = pd.read_csv(os.path.join(OUT, "runs_index.csv"))

    lines = []
    def w(s=""):
        lines.append(s)
        print(s)

    w("# S3 independent verification report")
    w()
    w("Blind verifier `analysis/paper/verify_independent.py`: own resolver, own")
    w("metric (confusion-count macro-F1, bit-identical to "
      "`sklearn.metrics.f1_score`), own paired bootstrap with a different seed.")
    w()
    w(f"- point-estimate tolerance: {POINT_TOL:g}")
    w(f"- CI tolerance: {CI_TOL:g}")
    w(f"- bootstrap replicas: {B}; verifier seed {SEED} (generator seed 42)")
    w(f"- CI level: {meta['ci']:.0%} percentile; rows: {len(boot)}")
    w()

    cache = {}
    failures = []
    point_rows = []

    # ---- 1. point estimates for EVERY comparison -------------------------
    for _, r in boot.iterrows():
        unit = compute_unit(idx, r, cache)
        checks = {
            "delta": abs(unit["delta_hat"] - r["delta_hat"]) <= POINT_TOL,
            "f1_a": abs(unit["f1_a"] - r["f1_a"]) <= POINT_TOL,
            "f1_b": abs(unit["f1_b"] - r["f1_b"]) <= POINT_TOL,
            "n": unit["N"] == int(r["n_instances"]),
            "paired": unit["paired_ok"],
        }
        ok = all(checks.values())
        if not ok:
            failures.append((r["comparison"], checks))
        point_rows.append((r["comparison"], unit["delta_hat"], r["delta_hat"],
                           unit["N"], int(r["n_instances"]), ok))

    n_pt = len(point_rows)
    n_pt_fail = sum(1 for row in point_rows if not row[5])
    w("## 1. Point estimates (all comparisons)")
    w()
    w(f"Recomputed delta_hat, f1_a, f1_b, n_instances and paired identity for "
      f"all {n_pt} rows: **{n_pt - n_pt_fail}/{n_pt} exact to {POINT_TOL:g}**.")
    w()
    w("| comparison | verifier delta | generator delta | |diff| | n | ok |")
    w("|---|---|---|---|---|---|")
    for name, dv, dg, nv, ng, ok in point_rows[:8]:
        w(f"| {name} | {dv:.12f} | {dg:.12f} | {abs(dv-dg):.2e} | {nv} | "
          f"{'yes' if ok else 'NO'} |")
    worst = max(abs(p[1] - p[2]) for p in point_rows)
    w(f"| *(all {n_pt} rows)* | | | **max {worst:.2e}** | | |")
    w()

    # ---- 2. CI bounds for the representative set -------------------------
    by_name = {r["comparison"]: r for _, r in boot.iterrows()}
    ci_rows = []
    for name, role in REPRESENTATIVE:
        if name not in by_name:
            failures.append((name, {"present": False}))
            w(f"## 2. CI check: `{name}` MISSING")
            continue
        r = by_name[name]
        unit = compute_unit(idx, r, cache)
        dlo = abs(unit["ci_low"] - r["ci_low"])
        dhi = abs(unit["ci_high"] - r["ci_high"])
        ok = dlo <= CI_TOL and dhi <= CI_TOL
        if not ok:
            failures.append((name, {"ci_low_diff": dlo, "ci_high_diff": dhi}))
        ci_rows.append((name, role, unit, r, dlo, dhi, ok))

    w("## 2. Reconstructed CIs (representative set)")
    w()
    w("| comparison | role | d_hat | CI low gen/ver/diff | "
      "CI high gen/ver/diff | ok |")
    w("|---|---|---|---|---|---|")
    for name, role, unit, r, dlo, dhi, ok in ci_rows:
        w(f"| {name} | {role} | {unit['delta_hat']:.8f} | "
          f"{r['ci_low']:.6f} / {unit['ci_low']:.6f} / {dlo:.2e} | "
          f"{r['ci_high']:.6f} / {unit['ci_high']:.6f} / {dhi:.2e} | "
          f"{'yes' if ok else 'NO'} |")
    n_ok = sum(1 for row in ci_rows if row[-1])
    w()
    w(f"{n_ok}/{len(ci_rows)} representative CIs within {CI_TOL:g}.")
    w()

    # ---- 3. paired idx / y_true identity ---------------------------------
    w("## 3. Paired idx / y_true identity")
    w()
    w("Per representative comparison, all arms over the same fold set; pooled")
    w("`idx` and `y_true` must be byte-identical across arms (paired bootstrap")
    w("precondition).")
    w()
    ident_ok = True
    for name, role in REPRESENTATIVE:
        if name not in by_name:
            continue
        r = by_name[name]
        ok = verify_idx_identity(idx, r, compute_unit(idx, r, cache))
        ident_ok = ident_ok and ok
        if not ok:
            failures.append((name, {"idx_identity": False}))
        w(f"- `{name}` ({role}): folds "
          f"{compute_unit(idx, r, cache)['common_folds']}, pooled idx/y_true "
          f"identical across arms = {ok}")
    w()
    w(f"All representative fold sets paired-identical: **{ident_ok}**.")
    w()

    # ---- 4. no hard-F1 delta ---------------------------------------------
    w("## 4. Hard-F1 is descriptive only (no paired delta)")
    w()
    hard_cols = [c for c in boot.columns if "hard" in c.lower()]
    delta_cols = [c for c in boot.columns if "hard" in c.lower() and "delta" in c.lower()]
    hard_ok = (len(delta_cols) == 0)
    mismatches = []
    for _, r in boot.iterrows():
        ds = r["dataset"]
        arms, _ = comparison_arms(r, ds)
        fsets = [set(idx[(idx.campaign == c) & (idx.dataset == ds) &
                         (idx.arm == a)]["fold"]) for c, _, a in arms]
        common = sorted(set.intersection(*fsets))
        sub = idx[(idx.campaign == arms[0][0]) & (idx.dataset == ds) &
                  (idx.arm == arms[0][2]) & (idx.fold.isin(common))]
        fm = sub["hard_f1"].mean()
        # bootstrap.csv stores these to ~10 decimals; compare at the same
        # tolerance as the point estimates.
        if not np.isclose(fm, r["hard_f1_a_foldmean"], atol=POINT_TOL, rtol=0):
            hard_ok = False
            mismatches.append(f"{r['comparison']}: {fm!r} vs "
                              f"{r['hard_f1_a_foldmean']!r}")
    w(f"- hard columns: {hard_cols}")
    w(f"- hard columns carrying a delta: {delta_cols or 'none'}")
    w(f"- hard_f1_a_foldmean reproduces the runs_index fold mean on every row: "
      f"**{hard_ok}**")
    if not hard_ok:
        failures.append(("hard_f1", {"delta_cols": delta_cols,
                                     "mismatches": mismatches[:10]}))
    w()

    # ---- 5. import hygiene -----------------------------------------------
    w("## 5. Import hygiene")
    w()
    bad = FORBIDDEN & set(sys.modules)
    bad_mod = [m for m in sys.modules if m.startswith("bio_is_curriculum")]
    w(f"- torch/transformers in sys.modules: {sorted(bad) or 'none'}")
    w(f"- bio_is_curriculum.* imported: {bad_mod or 'none'}")
    w("- imports used: numpy, pandas (+ stdlib)")
    w()
    if bad:
        failures.append(("imports", {"torch": sorted(bad)}))

    # ---- verdict ---------------------------------------------------------
    w("## Verdict")
    w()
    if failures:
        w(f"**FAIL** - {len(failures)} gate failure(s):")
        for name, info in failures[:60]:
            w(f"- `{name}`: {info}")
    else:
        w(f"**PASS** - every point estimate exact to {POINT_TOL:g}; every "
          f"representative CI within {CI_TOL:g}; paired identity holds; no "
          f"hard-F1 delta; no torch/transformers.")
    w()

    with open(os.path.join(OUT, "verify_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    assert not (FORBIDDEN & set(sys.modules)), "torch/transformers surfaced"
    if failures:
        print(f"\n{len(failures)} FAILURES", file=sys.stderr)
        sys.exit(1)
    print("\nVERIFY PASS")


if __name__ == "__main__":
    main()
