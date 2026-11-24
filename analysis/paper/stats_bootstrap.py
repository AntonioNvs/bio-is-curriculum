#!/usr/bin/env python
"""S2 - paired pooled-instance bootstrap for every planned comparison.

Read-only post-processing of the already-synced ``runpod_results/`` tree, driven
entirely by the S1 index (``analysis/paper/out/runs_index.csv``).  Nothing here
trains, fits or evaluates a model: it reads the saved per-fold
``predictions_test.csv`` files and recomputes macro-F1 on CPU.

Protocol (paper/acl_latex_new_experiments_plan.md, section 2):
  * Paired bootstrap over the *pooled* test instances of the folds.
  * B = 10,000, seed = 42, two-sided 90% percentile CI.
  * Resample indices are SHARED within a resampling unit (one dataset x one
    fold set), so every comparison - and the 4-arm interaction - is evaluated on
    exactly the same resamples.  Y5 comparisons (3 folds) form their own unit so
    their noise floor is measured on the Y5 replicate, not the 5-fold Y1 one.
  * Delta = MacroF1(A) - MacroF1(B).
  * NI: CI_low >= -delta.  Equivalence: CI subset of [-delta, +delta].
  * Interaction I = F(SPURS) - F(IS) - F(CL) + F(Raw) on the shared resamples.
  * One-sided bootstrap p-values (needed by Holm):
        p_lower = mean(Delta*_b <= -delta)   (evidence A < B beyond the margin)
        p_upper = mean(Delta*_b >= +delta)   (evidence A > B beyond the margin)
  * Hard F1 (V12) is descriptive only and is NEVER paired: per-fold hard F1 is
    recomputed from each model's own entropy slice and reported as a fold mean.

Outputs (analysis/paper/out/):
  bootstrap.csv       one row per comparison
  bootstrap_meta.json metadata (B, seed, CI, delta, dataset -> pooled sizes)

Guards: asserts torch / transformers are absent from ``sys.modules`` at start and
end, and never imports ``bio_is_curriculum`` or ``bio_is_curriculum.results``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- hard no-torch / no-transformers guard (checked again at exit) -----------
for _mod in ("torch", "transformers"):
    assert _mod not in sys.modules, f"{_mod} must not be imported by this script"

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Pre-committed protocol constants.
B_BOOT = 10_000
SEED = 42
CI = 0.90
DELTA = 0.01
HARD_SLICE_QUANTILE = 0.80  # top 20% by the model's own predictive entropy (V12)
# Resamples are generated and consumed in blocks so the full (B, n) matrix is
# never materialised (Yelp pools 335k instances -> 27 GB dense).  The block size
# is a fixed constant, so the RNG stream and hence the result stay deterministic.
RESAMPLE_BLOCK = 128

Y2_DATASETS = ("agnews", "books", "reuters90")
Y3_DATASETS = ("agnews", "books", "reuters90")
Y4_DATASETS = ("agnews", "books", "reuters90")
Y5_DATASETS = ("books", "reuters90")
ALL_DATASETS = ("yelp_2013", "agnews", "dblp", "books", "reuters90")

Y4_ARM_BY_DATASET = {
    "agnews": "raw_rand0.6831",
    "books": "raw_rand0.6596",
    "reuters90": "raw_rand0.637",
}
BETAS = ("0.1", "0.3", "0.5")
THETAS = ("0.0", "0.2", "0.4")


# --------------------------------------------------------------------------- #
# Comparison plan (section 2 + S2 text)
# --------------------------------------------------------------------------- #


def build_comparisons() -> list[dict]:
    c: list[dict] = []
    for ds in ALL_DATASETS:
        c.append(dict(comparison=f"SPURS-Raw/{ds}", kind="paired", gate="G-A",
                      a=("Y1", ds, "is_cl"), b=("Y1", ds, "raw"),
                      role_a="SPURS", role_b="Raw"))
        c.append(dict(comparison=f"IS-Raw/{ds}", kind="paired", gate="",
                      a=("Y1", ds, "is"), b=("Y1", ds, "raw"),
                      role_a="IS", role_b="Raw"))
        c.append(dict(comparison=f"CL-Raw/{ds}", kind="paired", gate="",
                      a=("Y1", ds, "cl"), b=("Y1", ds, "raw"),
                      role_a="CL", role_b="Raw"))
        c.append(dict(comparison=f"I/{ds}", kind="interaction", gate="G-G",
                      a=("Y1", ds, "is_cl"), b=("Y1", ds, "raw"),
                      role_a="Interaction I", role_b="Raw",
                      terms=dict(spurs=("Y1", ds, "is_cl"), is_=("Y1", ds, "is"),
                                 cl=("Y1", ds, "cl"), raw=("Y1", ds, "raw"))))
    for ds in Y2_DATASETS:
        c.append(dict(comparison=f"SPURS-Raw@2/{ds}", kind="paired", gate="G-F",
                      a=("Y1", ds, "is_cl"), b=("Y2", ds, "raw_ep2"),
                      role_a="SPURS", role_b="Raw@2"))
        c.append(dict(comparison=f"SPURS-Raw@4/{ds}", kind="paired", gate="G-F",
                      a=("Y1", ds, "is_cl"), b=("Y2", ds, "raw_ep4"),
                      role_a="SPURS", role_b="Raw@4"))
    for ds in Y4_DATASETS:
        c.append(dict(comparison=f"IS-Random@ret/{ds}", kind="paired", gate="G-E",
                      a=("Y1", ds, "is"), b=("Y4", ds, Y4_ARM_BY_DATASET[ds]),
                      role_a="IS", role_b="Random@ret"))
    for ds in Y3_DATASETS:
        c.append(dict(comparison=f"RandomPhases-Composite/{ds}", kind="paired", gate="G-C",
                      a=("Y3", ds, "sel_random_phases"), b=("Y3", ds, "sel_composite"),
                      role_a="RandomPhases", role_b="Composite"))
        c.append(dict(comparison=f"MarginOnly-Composite/{ds}", kind="paired", gate="G-D",
                      a=("Y3", ds, "sel_margin_only"), b=("Y3", ds, "sel_composite"),
                      role_a="MarginOnly", role_b="Composite"))
        c.append(dict(comparison=f"EntropyOnly-Composite/{ds}", kind="paired", gate="G-D",
                      a=("Y3", ds, "sel_entropy_only"), b=("Y3", ds, "sel_composite"),
                      role_a="EntropyOnly", role_b="Composite"))
    for ds in Y5_DATASETS:
        # G-H: theta>0 vs theta=0 at each beta (primary, Books).
        for beta in BETAS:
            c.append(dict(comparison=f"theta0.2-theta0.0/beta{beta}/{ds}", kind="paired",
                          gate="G-H", a=("Y5", ds, f"sweep_b{beta}_t0.2"),
                          b=("Y5", ds, f"sweep_b{beta}_t0.0"),
                          role_a="theta=0.2", role_b="theta=0.0"))
            c.append(dict(comparison=f"theta0.4-theta0.0/beta{beta}/{ds}", kind="paired",
                          gate="G-H", a=("Y5", ds, f"sweep_b{beta}_t0.4"),
                          b=("Y5", ds, f"sweep_b{beta}_t0.0"),
                          role_a="theta=0.4", role_b="theta=0.0"))
        # beta sweep (secondary).
        for theta in THETAS:
            c.append(dict(comparison=f"beta0.3-beta0.1/theta{theta}/{ds}", kind="paired",
                          gate="", a=("Y5", ds, f"sweep_b0.3_t{theta}"),
                          b=("Y5", ds, f"sweep_b0.1_t{theta}"),
                          role_a="beta=0.3", role_b="beta=0.1"))
            c.append(dict(comparison=f"beta0.5-beta0.1/theta{theta}/{ds}", kind="paired",
                          gate="", a=("Y5", ds, f"sweep_b0.5_t{theta}"),
                          b=("Y5", ds, f"sweep_b0.1_t{theta}"),
                          role_a="beta=0.5", role_b="beta=0.1"))
        # default grid cell vs the Y1 IS reference (freezes the Y5 noise floor).
        c.append(dict(comparison=f"default-IS/{ds}", kind="paired", gate="G-H",
                      a=("Y5", ds, "sweep_b0.3_t0.2"), b=("Y1", ds, "is"),
                      role_a="default", role_b="IS"))
    return c


# --------------------------------------------------------------------------- #
# Prediction loading (S1 index -> pooled per-fold arrays)
# --------------------------------------------------------------------------- #


def fold_map(index: pd.DataFrame) -> dict[tuple[str, str, str, int], str]:
    out: dict[tuple[str, str, str, int], str] = {}
    for r in index.itertuples(index=False):
        out[(r.campaign, r.dataset, r.arm, int(r.fold))] = r.dir
    return out


class DatasetBuilder:
    """Loads and pools the arms of one dataset for the shared-resample bootstrap."""

    def __init__(self, dataset: str, arms: list[tuple[str, str, str]], paths, folds):
        self.dataset = dataset
        self.folds = list(folds)
        self.y_true: list[np.ndarray] = []
        self.arms: dict[tuple[str, str, str], list[np.ndarray]] = {}
        self.hard: dict[tuple[str, str, str], list[float]] = {}
        for arm in arms:
            self.arms[arm] = []
        for fold in self.folds:
            fold_true = None
            for arm in arms:
                key = (arm[0], arm[1], arm[2], fold)
                if key not in paths:
                    raise SystemExit(f"BLOCKED: missing index cell {key}")
                pred = pd.read_csv(Path(paths[key]) / "predictions_test.csv")
                pred = pred.sort_values("idx").reset_index(drop=True)
                yt = pred["y_true"].to_numpy(dtype=np.int64)
                yp = pred["y_pred"].to_numpy(dtype=np.int64)
                if fold_true is None:
                    fold_true = yt
                elif not np.array_equal(fold_true, yt):
                    raise SystemExit(
                        f"BLOCKED: y_true mismatch across arms of {self.dataset} fold {fold}")
                self.arms[arm].append(yp)
                # V12 hard slice: top 20% by THIS model's own predictive entropy.
                ent = pred["pred_entropy"].to_numpy(dtype=float)
                thr = np.quantile(ent, HARD_SLICE_QUANTILE)
                mask = ent >= thr
                self.hard.setdefault(arm, []).append(
                    float(f1_score(yt[mask], yp[mask], average="macro")))
            self.y_true.append(fold_true)

        self.y_true_c = np.concatenate(self.y_true)
        self.labels = np.unique(self.y_true_c)
        self.k = len(self.labels)
        self.t = _map_labels(self.y_true_c, self.labels).astype(np.int64)
        self.n = int(self.t.shape[0])
        self.codes: dict[tuple[str, str, str], np.ndarray] = {}
        for arm in arms:
            pooled = _map_labels(np.concatenate(self.arms[arm]), self.labels).astype(np.int64)
            # joint class code (t*k + p) in int32: k <= 90 so codes < 2**31.
            self.codes[arm] = (self.t * self.k + pooled).astype(np.int32)
        self.f1_point: dict[tuple[str, str, str], float] = {}
        for arm in arms:
            self.f1_point[arm] = _macro_f1_from_code(self.codes[arm], self.k)

    def arm_foldmean_hard(self, arm) -> float:
        return float(np.mean(self.hard[arm]))


def _map_labels(y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    index = {int(c): i for i, c in enumerate(labels)}
    return np.array([index[int(v)] for v in y], dtype=np.int64)


def _macro_f1_from_code(code: np.ndarray, k: int) -> float:
    """Macro-F1 from joint (t*k + p) codes, exactly as equivalence.py does."""
    cm = np.bincount(code, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(prec), where=denom > 0)
    return float(np.mean(f1))


def _resample_f1_series(code: np.ndarray, k: int, idx_block: np.ndarray) -> np.ndarray:
    """Macro-F1 for each resample row in ``idx_block`` (flattened joint-code codes).

    Each resample row is offset by ``row * k * k`` before the single ``bincount``
    so that the (b, k, k) confusion matrices do not collide.
    """
    kk = k * k
    b, n = idx_block.shape
    assert b * kk < 2 ** 31, "block too large for int32 codes"
    sel = code[idx_block]  # (b, n)
    sel = sel + (np.arange(b, dtype=sel.dtype) * kk)[:, None]
    counts = np.bincount(sel.reshape(-1), minlength=b * kk).reshape(b, kk).astype(float)
    cm = counts.reshape(b, k, k)
    tp = cm[:, np.arange(k), np.arange(k)]
    fp = cm.sum(axis=1) - tp
    fn = cm.sum(axis=2) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(prec), where=denom > 0)
    return f1.mean(axis=1)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def bootstrap_dataset(builder: DatasetBuilder, comparisons: list[dict],
                      rng: np.random.Generator) -> dict[str, dict]:
    """One shared resample stream; every comparison of this dataset reuses it.

    Resamples are drawn and consumed in blocks of ``RESAMPLE_BLOCK`` so the
    (B, n) index matrix is never materialised.  The RNG is drawn once in blocks
    of a fixed size, so the sequence - and the result - is deterministic.
    """
    n = builder.n
    k = builder.k

    needed: list[tuple[str, str, str]] = []
    for cmp in comparisons:
        if cmp["kind"] == "interaction":
            arm_list = list(cmp["terms"].values())
        else:
            arm_list = [cmp["a"], cmp["b"]]
        for arm in arm_list:
            if arm not in needed:
                needed.append(arm)

    # Accumulate the per-resample F1 of every needed arm across blocks.
    stored: dict[tuple[str, str, str], np.ndarray] = {
        arm: np.empty(B_BOOT, dtype=float) for arm in needed
    }
    pos = 0
    while pos < B_BOOT:
        b = min(RESAMPLE_BLOCK, B_BOOT - pos)
        idx_block = rng.integers(0, n, size=(b, n), dtype=np.int64)
        for arm in needed:
            stored[arm][pos:pos + b] = _resample_f1_series(builder.codes[arm], k, idx_block)
        pos += b

    out: dict[str, dict] = {}
    for cmp in comparisons:
        if cmp["kind"] == "interaction":
            s = stored[cmp["terms"]["spurs"]]
            i = stored[cmp["terms"]["is_"]]
            cl = stored[cmp["terms"]["cl"]]
            raw = stored[cmp["terms"]["raw"]]
            diffs = s - i - cl + raw
            point = (builder.f1_point[cmp["terms"]["spurs"]]
                     - builder.f1_point[cmp["terms"]["is_"]]
                     - builder.f1_point[cmp["terms"]["cl"]]
                     + builder.f1_point[cmp["terms"]["raw"]])
            f1_a = builder.f1_point[cmp["terms"]["spurs"]]
            f1_b = builder.f1_point[cmp["terms"]["raw"]]
            hard_a = builder.arm_foldmean_hard(cmp["terms"]["spurs"])
            hard_b = builder.arm_foldmean_hard(cmp["terms"]["raw"])
        else:
            diffs = stored[cmp["a"]] - stored[cmp["b"]]
            point = builder.f1_point[cmp["a"]] - builder.f1_point[cmp["b"]]
            f1_a = builder.f1_point[cmp["a"]]
            f1_b = builder.f1_point[cmp["b"]]
            hard_a = builder.arm_foldmean_hard(cmp["a"])
            hard_b = builder.arm_foldmean_hard(cmp["b"])

        ci_low = float(np.percentile(diffs, 100.0 * (1.0 - CI) / 2.0))
        ci_high = float(np.percentile(diffs, 100.0 * (1.0 + CI) / 2.0))
        p_lower = float(np.mean(diffs <= -DELTA))
        p_upper = float(np.mean(diffs >= DELTA))
        ni_pass = bool(ci_low >= -DELTA)
        equiv_pass = bool(ci_low >= -DELTA and ci_high <= DELTA)
        if equiv_pass:
            equiv_status = "equivalent"
        elif ni_pass:
            equiv_status = "non-inferior, not equivalent"
        else:
            equiv_status = "not established"
        out[cmp["comparison"]] = dict(
            diffs=diffs,
            point=float(point),
            ci_low=ci_low,
            ci_high=ci_high,
            p_lower=p_lower,
            p_upper=p_upper,
            ni_pass=ni_pass,
            equivalence_pass=equiv_pass,
            equivalence_status=equiv_status,
            f1_a=float(f1_a),
            f1_b=float(f1_b),
            hard_a=float(hard_a),
            hard_b=float(hard_b),
            ci_excludes_zero=bool(ci_low > 0.0 or ci_high < 0.0),
        )
    return out


# --------------------------------------------------------------------------- #
# V7 replicate noise floor (pooled |dF1| per dataset)
# --------------------------------------------------------------------------- #

SEL_COMPOSITE_DS = ("agnews", "books", "reuters90")


def compute_noise_floor(ds: str, f1map: dict) -> tuple[float, str]:
    """V7 replicate noise floor for a comparison unit, as (value, source).

    The floor is the pooled |MacroF1(replicate) - MacroF1(reference)| of a
    same-selection, same-config rerun (V6/V7).  A unit whose arms include Y5
    cells uses the Y5-default-vs-Y1-IS replicate; otherwise the
    Y3-composite-vs-Y1-SPURS replicate.  Datasets without a replicate (dblp,
    yelp_2013) have no floor and return (nan, "not available").
    """
    if any(a[0] == "Y5" for a in f1map):
        a, b = ("Y5", ds, "sweep_b0.3_t0.2"), ("Y1", ds, "is")
        src = "Y5default_vs_Y1IS"
    elif ds in SEL_COMPOSITE_DS:
        a, b = ("Y3", ds, "sel_composite"), ("Y1", ds, "is_cl")
        src = "Y3composite_vs_Y1SPURS"
    else:
        return float("nan"), "not available"
    if a in f1map and b in f1map:
        return float(abs(f1map[a] - f1map[b])), src
    return float("nan"), "not available"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    global B_BOOT, SEED, CI, DELTA

    ap = argparse.ArgumentParser(description="S2 paired pooled-instance bootstrap.")
    ap.add_argument("--index", default="analysis/paper/out/runs_index.csv")
    ap.add_argument("--out-dir", default="analysis/paper/out")
    ap.add_argument("--n-bootstrap", type=int, default=B_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ci", type=float, default=CI)
    ap.add_argument("--delta", type=float, default=DELTA)
    args = ap.parse_args()

    B_BOOT, SEED, CI, DELTA = args.n_bootstrap, args.seed, args.ci, args.delta

    index = pd.read_csv(args.index)
    paths = fold_map(index)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparisons = build_comparisons()

    # Group comparisons into resampling units.  Arms with the same fold set of
    # one dataset must share the resample stream (e.g. all Y1 arms have 5 folds).
    # Y5 arms have 3 folds, so their comparisons form a separate unit; their
    # noise floor is computed on the pooled 3-fold Y5 instances, not the Y1 ones.
    def arm_of(cmp, which):
        if cmp["kind"] == "interaction":
            return cmp["terms"]["spurs" if which == "a" else "raw"]
        return cmp[which]

    groups: dict[tuple[str, tuple[int, ...]], list[dict]] = {}
    for cmp in comparisons:
        if cmp["kind"] == "interaction":
            unit_arms = list(cmp["terms"].values())
        else:
            unit_arms = [cmp["a"], cmp["b"]]
        ds = unit_arms[0][1]
        arm_folds = [{k[3] for k in paths if k[:3] == tuple(a)} for a in unit_arms]
        folds = tuple(sorted(set.intersection(*arm_folds)))
        if not folds:
            raise SystemExit(f"BLOCKED: no shared folds for {cmp['comparison']}")
        groups.setdefault((ds, folds), []).append(cmp)

    rows: list[dict] = []
    meta: dict = dict(
        n_bootstrap=args.n_bootstrap, seed=args.seed, ci=args.ci, delta=args.delta,
        protocol="paired pooled-instance percentile bootstrap; resamples shared "
                 "across the arms of a resampling unit (dataset x fold-set)",
        p_value_definitions=dict(
            p_lower="mean(Delta*_b <= -delta) [evidence A < B beyond margin]",
            p_upper="mean(Delta*_b >= +delta) [evidence A > B beyond margin]",
        ),
        noise_floor_definition="pooled |MacroF1(replicate) - MacroF1(reference)| "
                               "for a same-selection, same-config rerun (V7), "
                               "computed on the fold set of the comparison unit",
        hard_f1_note="hard_f1_a_foldmean / hard_f1_b_foldmean are descriptive "
                     "fold means of the per-model top-20%-entropy macro-F1 (V12); "
                     "they are never paired and carry no delta.",
        datasets={},
        comparisons=[],
    )

    # V7 reference arms we always load, per unit.
    v7_arms = [("Y1", "is_cl"), ("Y3", "sel_composite"),
               ("Y1", "is"), ("Y5", "sweep_b0.3_t0.2")]

    for (ds, folds), unit_cmps in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        arms: set[tuple[str, str, str]] = set()
        for cmp in unit_cmps:
            if cmp["kind"] == "interaction":
                arms.update(cmp["terms"].values())
            else:
                arms.add(cmp["a"])
                arms.add(cmp["b"])
        # attach the V7 arms that actually exist for this dataset + fold set
        for camp, arm in v7_arms:
            if all((camp, ds, arm, f) in paths for f in folds):
                arms.add((camp, ds, arm))
        arms = sorted(arms)
        unit_key = f"{ds}|folds={'-'.join(map(str, folds))}"

        builder = DatasetBuilder(ds, arms, paths, folds)
        rng = np.random.default_rng(SEED)
        results = bootstrap_dataset(builder, unit_cmps, rng)

        f1map = {a: builder.f1_point[a] for a in arms}
        floor, floor_src = compute_noise_floor(ds, f1map)

        meta["datasets"][unit_key] = dict(
            dataset=ds, n_instances=builder.n, n_folds=len(folds), n_classes=builder.k,
            folds=list(folds), arms=[list(a) for a in arms],
            noise_floor_value=(None if np.isnan(floor) else float(floor)),
            noise_floor_source=floor_src,
        )

        for cmp in unit_cmps:
            r = results[cmp["comparison"]]
            below = bool(abs(r["point"]) < floor) if not np.isnan(floor) else False
            a_arm = arm_of(cmp, "a")
            b_arm = arm_of(cmp, "b")
            row = dict(
                comparison=cmp["comparison"],
                kind=cmp["kind"],
                gate=cmp["gate"],
                dataset=ds,
                campaign_a=a_arm[0],
                arm_a=a_arm[2],
                role_a=cmp["role_a"],
                campaign_b=b_arm[0],
                arm_b=b_arm[2],
                role_b=cmp["role_b"],
                n_instances=builder.n,
                n_folds=len(folds),
                n_classes=builder.k,
                f1_a=r["f1_a"],
                f1_b=r["f1_b"],
                delta_hat=r["point"],
                ci_low=r["ci_low"],
                ci_high=r["ci_high"],
                ci_level=CI,
                delta=DELTA,
                ni_pass=r["ni_pass"],
                equivalence_pass=r["equivalence_pass"],
                equivalence_status=r["equivalence_status"],
                ci_excludes_zero=r["ci_excludes_zero"],
                p_lower=r["p_lower"],
                p_upper=r["p_upper"],
                # pre-committed one-sided p for Holms: p_d = mean(Delta*_b <= -delta)
                p_d=r["p_lower"],
                noise_floor=floor,
                below_noise=below,
                hard_f1_a_foldmean=r["hard_a"],
                hard_f1_b_foldmean=r["hard_b"],
                n_bootstrap=B_BOOT,
                seed=SEED,
            )
            rows.append(row)
            meta["comparisons"].append(dict(
                comparison=cmp["comparison"], dataset=ds, gate=cmp["gate"],
                n_instances=builder.n, n_folds=len(folds),
                delta_hat=r["point"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                noise_floor=(None if np.isnan(floor) else float(floor)),
                below_noise=below,
            ))

    df = pd.DataFrame(rows)
    csv_path = out_dir / "bootstrap.csv"
    df.to_csv(csv_path, index=False, float_format="%.10g")

    meta["n_comparisons"] = len(df)
    meta["config"] = dict(index=str(args.index), n_bootstrap=args.n_bootstrap,
                          seed=args.seed, ci=args.ci, delta=args.delta)
    (out_dir / "bootstrap_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # --- schema / finiteness validation -------------------------------------
    required = ["comparison", "kind", "gate", "dataset", "n_instances", "n_folds",
                "n_classes", "delta_hat", "ci_low", "ci_high", "ci_level", "delta",
                "ni_pass", "equivalence_pass", "ci_excludes_zero", "p_lower",
                "p_upper", "p_d", "noise_floor", "below_noise", "n_bootstrap", "seed"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"FAIL: bootstrap.csv missing columns {missing}")
    finite_cols = ["delta_hat", "ci_low", "ci_high", "ci_level", "delta",
                   "p_lower", "p_upper", "p_d",
                   "hard_f1_a_foldmean", "hard_f1_b_foldmean"]
    bad = {c: int((~np.isfinite(df[c])).sum()) for c in finite_cols}
    bad = {c: n for c, n in bad.items() if n}
    if bad:
        raise SystemExit(f"FAIL: non-finite values in {bad}")
    if (df["ci_low"] > df["ci_high"]).any():
        raise SystemExit("FAIL: ci_low > ci_high in some row")
    if not (df["n_bootstrap"] == B_BOOT).all() or not (df["seed"] == SEED).all():
        raise SystemExit("FAIL: n_bootstrap / seed not propagated")
    # a noise floor is required wherever a V7 replicate exists (agnews, books,
    # reuters90); dblp/yelp_2013 have no same-config rerun and are legitimately NaN.
    has_replicate = df["dataset"].isin(["agnews", "books", "reuters90"])
    if np.isnan(df.loc[has_replicate, "noise_floor"]).any():
        raise SystemExit("FAIL: missing noise floor for a dataset that has a replicate")

    for mod in ("torch", "transformers"):
        assert mod not in sys.modules, f"{mod} leaked into sys.modules"

    print(f"wrote {csv_path} ({len(df)} rows) and {out_dir / 'bootstrap_meta.json'}")
    print(f"[INFO] B={B_BOOT} seed={SEED} ci={CI} delta={DELTA}")
    for unit, info in sorted(meta["datasets"].items()):
        print(f"[INFO] {unit}: n={info['n_instances']} folds={info['n_folds']} "
              f"classes={info['n_classes']} noise_floor={info['noise_floor_value']} "
              f"({info['noise_floor_source']})")
    print(f"[INFO] below-noise comparisons: {int(df['below_noise'].sum())}/{len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
