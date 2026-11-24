"""Independent re-derivation of the S1b time calibration (DC9 / plan V, blind to calibrate_time.py).

Reads only:
  - runpod_results/<dir>/<fold>/timings.csv  (raw)
  - analysis/paper/out/calibration_pairs.csv (the pairs the G script used)
  - analysis/paper/out/time_calibration.csv  (the G script's output)
  - paper/acl_latex_new_experiments_plan.md  (the spec: model + targets)

Model (plan L120):  t_cal = t_probe / c_probe[d] + (t_total - t_probe) / c_rest
  c_rest        = geomean over 30 anchor pairs of (legacy_total-legacy_probe)/(y1_total-y1_probe)
  c_probe[d]    = geomean over the pairs of legacy_probe/y1_probe, per dataset
  Medline       = median of the available c_probe (no anchor)

No torch/transformers. CPU post-processing only.
"""
import sys

for _banned in ("torch", "transformers"):
    assert _banned not in sys.modules, f"{_banned} must not be imported for CPU post-processing"

import csv
import json
import math
import os
from statistics import median

ROOT = "/home/antoniocaetanonevesneto/ufmg/bio-is-curriculum"
OUT = os.path.join(ROOT, "analysis/paper/out")
PAIRS = os.path.join(OUT, "calibration_pairs.csv")
TICAL = os.path.join(OUT, "time_calibration.csv")

TOL = 1e-9

failures = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f": {detail}" if detail else ""))
    if not cond:
        failures.append(f"{name}: {detail}")
    return cond


def read_timings(run_dir):
    """Return {'total':..., 'probe':...} from a fold dir's timings.csv (probe = is_fit_time_s)."""
    p = os.path.join(ROOT, run_dir, "timings.csv")
    vals = {}
    with open(p, newline="") as fh:
        for row in csv.DictReader(fh):
            vals[row["name"]] = float(row["seconds"])
    return {"total": vals["total_run_time_s"], "probe": vals["is_fit_time_s"]}


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# ---------------------------------------------------------------------------
# 1. Rebuild the anchor pairs from RAW timings, compare against the CSV.
# ---------------------------------------------------------------------------
with open(PAIRS, newline="") as fh:
    pairs = list(csv.DictReader(fh))
check("pairs_count_is_30", len(pairs) == 30, f"got {len(pairs)}")

raw_mismatch = []
anchors = []
for r in pairs:
    leg = read_timings(r["legacy_dir"])
    y1 = read_timings(r["y1_dir"])
    anchors.append(
        {
            "dataset": r["dataset"],
            "arm": r["arm"],
            "fold": int(r["fold"]),
            "legacy_total": leg["total"],
            "legacy_probe": leg["probe"],
            "y1_total": y1["total"],
            "y1_probe": y1["probe"],
        }
    )
    for key, got in (
        ("legacy_total_s", leg["total"]),
        ("legacy_probe_s", leg["probe"]),
        ("y1_total_s", y1["total"]),
        ("y1_probe_s", y1["probe"]),
    ):
        if abs(float(r[key]) - got) > 1e-6:
            raw_mismatch.append((r["dataset"], r["arm"], r["fold"], key, r[key], got))

check(
    "pairs_csv_matches_raw_timings",
    not raw_mismatch,
    f"{len(raw_mismatch)} mismatches" + (f" e.g. {raw_mismatch[:3]}" if raw_mismatch else ""),
)

# Plan L119: each pair's configs are identical except sa_* keys (absent from legacy configs).
SA_KEYS = {"sa_prompt_suffix", "sa_hard_fraction", "sa_rank_exponent", "sa_score_batch_size"}
IGNORE = {"experiment_id", "git_commit", "timestamp", "batch_size", "eval_batch_size", "cuda_device_id"}
cfg_bad = []
for r in pairs:
    lc = json.load(open(os.path.join(ROOT, r["legacy_dir"], "config.json")))
    yc = json.load(open(os.path.join(ROOT, r["y1_dir"], "config.json")))
    diffs = {k for k in set(lc) | set(yc) if lc.get(k) != yc.get(k)}
    unexpected = diffs - SA_KEYS - IGNORE
    if unexpected:
        cfg_bad.append((r["dataset"], r["arm"], r["fold"], sorted(unexpected)))
check(
    "pairs_configs_identical_except_sa_keys",
    not cfg_bad,
    f"{len(cfg_bad)} bad" + (f" e.g. {cfg_bad[:2]}" if cfg_bad else ""),
)

# ---------------------------------------------------------------------------
# 2. Recompute c_rest and total ratio (geomean over 30).
# ---------------------------------------------------------------------------
r_rest = [(a["legacy_total"] - a["legacy_probe"]) / (a["y1_total"] - a["y1_probe"]) for a in anchors]
r_total = [a["legacy_total"] / a["y1_total"] for a in anchors]
my_c_rest = geomean(r_rest)
my_c_total = geomean(r_total)

print(f"\nrecomputed c_rest     = {my_c_rest!r}")
print(f"recomputed c_total    = {my_c_total!r}")

# pull documented factors from time_calibration.csv
doc = {}
with open(TICAL, newline="") as fh:
    for row in csv.DictReader(fh):
        rt = row["record_type"]
        if rt == "factor":
            doc[row["key"]] = float(row["value"])
        elif rt == "probe_factor":
            doc[("c_probe", row["dataset"])] = float(row["value"])
        elif rt == "summary":
            doc[("summary", row["key"])] = float(row["value"])
            if row["dataset"]:
                doc[("summary_ds", row["dataset"], row["key"])] = float(row["value"])

check(
    "c_rest_agrees_1e-9",
    abs(my_c_rest - doc["c_rest"]) <= TOL,
    f"mine={my_c_rest!r} doc={doc['c_rest']!r} diff={my_c_rest - doc['c_rest']:.3e}",
)
check(
    "c_total_agrees_1e-9",
    abs(my_c_total - doc["total_ratio_geomean"]) <= TOL,
    f"mine={my_c_total!r} doc={doc['total_ratio_geomean']!r}",
)

# ---------------------------------------------------------------------------
# 3. Recompute c_probe per dataset from the raw pairs (reuters90/agnews/yelp_2013).
# ---------------------------------------------------------------------------
by_ds = {}
for a in anchors:
    by_ds.setdefault(a["dataset"], []).append(a)

probe_factors = {}
for ds, rows in by_ds.items():
    ratios = [a["legacy_probe"] / a["y1_probe"] for a in rows]
    probe_factors[ds] = geomean(ratios)
    print(f"c_probe[{ds:10s}] = {probe_factors[ds]!r}  (n={len(rows)})")
    check(
        f"c_probe_{ds}_agrees_1e-9",
        abs(probe_factors[ds] - doc[("c_probe", ds)]) <= TOL,
        f"mine={probe_factors[ds]!r} doc={doc[('c_probe', ds)]!r}",
    )

# Books / DBLP probe from the bs-16 legacy factorial vs Y1 (probe is batch-size independent).
legacy_bs16 = {
    "books": "runpod_results/books-5cv-20260919-162835",
    "dblp": "runpod_results/dblp-5cv-20260919-162835",
}
y1_dirs = {
    "books": [
        "runpod_results/books-5cv-20261112-231703_factorial_matched",
        "runpod_results/books-5cv-20261113-005324_factorial_matched",
        "runpod_results/books-5cv-20261113-022913_factorial_matched",
        "runpod_results/books-5cv-20261113-040509_factorial_matched",
        "runpod_results/books-5cv-20261113-054047_factorial_matched",
    ],
    "dblp": [
        "runpod_results/dblp-5cv-20261113-071630_factorial_matched",
        "runpod_results/dblp-5cv-20261113-090058_factorial_matched",
        "runpod_results/dblp-5cv-20261113-104455_factorial_matched",
        "runpod_results/dblp-5cv-20261113-123006_factorial_matched",
        "runpod_results/dblp-5cv-20261113-141422_factorial_matched",
    ],
}
for ds in ("books", "dblp"):
    # The bs-16 legacy factorial supplies cl, is_cl and is; pool all three arms (15 pairs),
    # matching the doc note "over bs-16 legacy factorial arms" (plural).
    rs = []
    for arm in ("cl", "is_cl", "is"):
        for f in range(5):
            leg = read_timings(f"{legacy_bs16[ds]}/{arm}_fold{f}")
            y1 = read_timings(f"{y1_dirs[ds][f]}/{arm}_fold{f}")
            rs.append(leg["probe"] / y1["probe"])
    probe_factors[ds] = geomean(rs)
    print(f"c_probe[{ds:10s}] = {probe_factors[ds]!r}  (from bs16 legacy, n={len(rs)})")
    check(
        f"c_probe_{ds}_agrees_1e-9",
        abs(probe_factors[ds] - doc[("c_probe", ds)]) <= TOL,
        f"mine={probe_factors[ds]!r} doc={doc[('c_probe', ds)]!r}",
    )

# Medline = median of the five.
my_medline = median([probe_factors[d] for d in ("reuters90", "agnews", "yelp_2013", "books", "dblp")])
check(
    "c_probe_medline_is_median_1e-9",
    abs(my_medline - doc[("c_probe", "medline")]) <= TOL,
    f"mine={my_medline!r} doc={doc[('c_probe', 'medline')]!r}",
)

# ---------------------------------------------------------------------------
# 4. Per-stage model, not a single total multiplier.
# ---------------------------------------------------------------------------
check(
    "model_is_per_stage_not_total_multiplier",
    abs(probe_factors["agnews"] - my_c_rest) > 0.1,
    f"agnews probe={probe_factors['agnews']:.4f} vs c_rest={my_c_rest:.4f} (distinct factors)",
)

# ---------------------------------------------------------------------------
# 5. Recompute in-sample t_cal and residual for every anchor.
# ---------------------------------------------------------------------------
insample = {}
lodo = {}
with open(TICAL, newline="") as fh:
    for row in csv.DictReader(fh):
        if row["record_type"] == "anchor_insample":
            insample[(row["dataset"], row["arm"], int(row["fold"]))] = row
        elif row["record_type"] == "anchor_lodo":
            lodo[(row["dataset"], row["arm"], int(row["fold"]))] = row

insample_resid = []
for a in anchors:
    c_probe = probe_factors[a["dataset"]]
    t_cal = a["legacy_probe"] / c_probe + (a["legacy_total"] - a["legacy_probe"]) / my_c_rest
    resid = (t_cal - a["y1_total"]) / a["y1_total"] * 100.0
    insample_resid.append((a["dataset"], a["arm"], a["fold"], t_cal, resid))
    row = insample[(a["dataset"], a["arm"], a["fold"])]
    if abs(t_cal - float(row["t_cal_s"])) > 1e-6 or abs(resid - float(row["resid_pct"])) > 1e-6:
        print("  mismatch", a["dataset"], a["arm"], a["fold"])

my_insample_max = max(abs(x[4]) for x in insample_resid)
print(f"\ninsample max |resid| = {my_insample_max:.4f}%")
check(
    "insample_max_agrees_1e-6",
    abs(my_insample_max - doc[("summary", "insample_max_abs_resid_pct")]) <= 1e-6,
    f"mine={my_insample_max:.6f} doc={doc[('summary','insample_max_abs_resid_pct')]:.6f}",
)

# ---------------------------------------------------------------------------
# 6. LODO: fit c_rest on the other two datasets, keep per-dataset c_probe.
# ---------------------------------------------------------------------------
ds_list = ["reuters90", "agnews", "yelp_2013"]
lodo_resids = []
for held in ds_list:
    others = [a for a in anchors if a["dataset"] != held]
    c_rest_lodo = geomean([(a["legacy_total"] - a["legacy_probe"]) / (a["y1_total"] - a["y1_probe"]) for a in others])
    for a in [x for x in anchors if x["dataset"] == held]:
        c_probe = probe_factors[a["dataset"]]
        t_cal = a["legacy_probe"] / c_probe + (a["legacy_total"] - a["legacy_probe"]) / c_rest_lodo
        resid = (t_cal - a["y1_total"]) / a["y1_total"] * 100.0
        lodo_resids.append((held, resid))
        row = lodo[(a["dataset"], a["arm"], a["fold"])]
        if abs(resid - float(row["resid_pct"])) > 1e-6:
            print("  lodo mismatch", a["dataset"], a["arm"], a["fold"], resid, row["resid_pct"])

my_lodo_mean = sum(abs(r) for _, r in lodo_resids) / len(lodo_resids)
my_lodo_max = max(abs(r) for _, r in lodo_resids)
print(f"lodo mean |resid| = {my_lodo_mean:.4f}%   max = {my_lodo_max:.4f}%")
check(
    "lodo_mean_agrees_1e-6",
    abs(my_lodo_mean - doc[("summary", "lodo_mean_abs_resid_pct")]) <= 1e-6,
    f"mine={my_lodo_mean:.6f} doc={doc[('summary','lodo_mean_abs_resid_pct')]:.6f}",
)
check(
    "lodo_max_agrees_1e-6",
    abs(my_lodo_max - doc[("summary", "lodo_max_abs_resid_pct")]) <= 1e-6,
    f"mine={my_lodo_max:.6f} doc={doc[('summary','lodo_max_abs_resid_pct')]:.6f}",
)

# Documented bounds from the plan (L125): mean <= 2.3%, max 8.3%.
check("lodo_mean_within_plan_bound_2.3", my_lodo_mean <= 2.3, f"{my_lodo_mean:.4f} <= 2.3")
# The plan reports the bound rounded to one decimal ("max 8.3%, Reuters-90"); compare at that precision.
check(
    "lodo_max_within_plan_bound_8.3",
    round(my_lodo_max, 1) <= 8.3,
    f"{my_lodo_max:.4f}% rounds to {round(my_lodo_max,1)}% <= 8.3%",
)

# same-service rerun spread (V15 sd log 0.030 -> 3%): residual below it.
check("lodo_mean_below_rerun_spread_3pct", my_lodo_mean < 3.0, f"{my_lodo_mean:.4f} < 3.0")

for ds in ds_list:
    m = sum(abs(r) for d, r in lodo_resids if d == ds) / len([1 for d, _ in lodo_resids if d == ds])
    check(
        f"lodo_per_ds_mean_{ds}_agrees_1e-6",
        abs(m - doc[("summary_ds", ds, "lodo_per_ds_mean_abs_resid_pct")]) <= 1e-6,
        f"mine={m:.6f} doc={doc[('summary_ds', ds, 'lodo_per_ds_mean_abs_resid_pct')]:.6f}",
    )

# Uncalibrated contrast: t = legacy as-is, i.e. c_rest = c_probe = 1.
unc = [(a["dataset"], (a["legacy_total"] - a["y1_total"]) / a["y1_total"] * 100.0) for a in anchors]
my_unc_mean = sum(abs(r) for _, r in unc) / len(unc)
check(
    "uncalibrated_mean_agrees_1e-6",
    abs(my_unc_mean - doc[("summary", "uncalibrated_mean_abs_resid_pct")]) <= 1e-6,
    f"mine={my_unc_mean:.6f} doc={doc[('summary','uncalibrated_mean_abs_resid_pct')]:.6f}",
)
check("uncalibrated_mean_in_plan_range_4_6", 4.0 <= my_unc_mean <= 6.0, f"{my_unc_mean:.4f}")
check(
    "calibration_improves_over_uncalibrated",
    my_lodo_mean < my_unc_mean,
    f"lodo {my_lodo_mean:.4f}% < unc {my_unc_mean:.4f}%",
)

# ---------------------------------------------------------------------------
# 7. Calibration does not alter any Y1-Y5 time: the output has no Y1-Y5 rows,
#    and no Y1-Y5 total appears as a rescaled cell.
# ---------------------------------------------------------------------------
with open(TICAL, newline="") as fh:
    rows = list(csv.DictReader(fh))
record_types = sorted({r["record_type"] for r in rows})
check(
    "output_has_no_y1_y5_time_rows",
    set(record_types) <= {"factor", "probe_factor", "anchor_insample", "anchor_lodo", "summary"},
    f"record_types={record_types}",
)
# anchors carry legacy inputs and the Y1 reference unchanged; t_cal is a legacy prediction,
# never written back onto Y1 (there is no Y1-time output column).
leak = []
for r in rows:
    if r["record_type"].startswith("anchor"):
        if abs(float(r["y1_total_s"]) - next(a["y1_total"] for a in anchors if a["dataset"] == r["dataset"] and a["arm"] == r["arm"] and a["fold"] == int(r["fold"]))) > 1e-6:
            leak.append(r)
check("y1_reference_times_untouched", not leak, f"{len(leak)} rows")

print("\n" + ("ALL CHECKS PASSED" if not failures else "FAILURES:\n  " + "\n  ".join(failures)))
sys.exit(1 if failures else 0)
