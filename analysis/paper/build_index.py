#!/usr/bin/env python
"""S1 - build the matched-runs index and assert the DC1 invariants.

Read-only post-processing of the already-synced ``runpod_results/`` tree.
Nothing here trains, fits, or evaluates a model: it reads the saved
``config.json`` / ``phase_metrics.csv`` / ``timings.csv`` / ``predictions_test.csv``
/ ``instance_selection.json`` files and recomputes metrics on CPU with sklearn.

Outputs (analysis/paper/out/):
  runs_index.csv   one row per (campaign, dataset, arm, fold)
  invariants.log   DC1 I1..I11 verdicts

Design notes (see paper/acl_latex_new_experiments_plan.md):
  * V2: ``runpod_results/`` holds one (dataset, fold) shard per directory, so the
    stock resolvers (newest-dir-wins) must NOT be used.  This index merges folds
    across directories.
  * V3: a duplicate shard (AG News raw_ep* fold 1) is resolved to the newest
    *complete* directory; the dropped one is logged.
  * V4: the final phase is reported.  F1 is recomputed from ``predictions_test.csv``;
    aggregate F1 is NEVER read from ``summary.csv`` (asserted: that file is never
    opened).
  * V5/V6/V10 are recomputed here rather than hardcoded (they are re-asserted).

The script asserts ``torch``/``transformers`` are absent from ``sys.modules`` and
never imports ``bio_is_curriculum``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# --- hard no-torch / no-transformers guard (checked again at exit) -----------
for _mod in ("torch", "transformers"):
    assert _mod not in sys.modules, f"{_mod} must not be imported by this script"

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# --------------------------------------------------------------------------- #
# Loader helpers (BOM tolerant)
# --------------------------------------------------------------------------- #

OPENED_FILES: list[str] = []


def _read_json(path: Path) -> dict:
    OPENED_FILES.append(str(path))
    assert path.name != "summary.csv", "aggregate F1 must never be read"
    with open(path, encoding="utf-8-sig") as fh:
        return json.load(fh)


def _read_csv(path: Path) -> pd.DataFrame:
    OPENED_FILES.append(str(path))
    assert path.name != "summary.csv", "aggregate F1 must never be read"
    return pd.read_csv(path, encoding="utf-8-sig")


def _read_timings(path: Path) -> dict[str, float]:
    OPENED_FILES.append(str(path))
    out: dict[str, float] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0] != "name":
                out[row[0]] = float(row[1])
    return out


# --------------------------------------------------------------------------- #
# Spec (from the plan)
# --------------------------------------------------------------------------- #

# V8 dataset order (Table 2), resolved by descending training size.
DATASET_ORDER = ["yelp_2013", "agnews", "dblp", "books", "reuters90"]

# V5 measured retention 1-rho (Y1 IS, fold mean).
V5_RETENTION = {
    "yelp_2013": 0.618,
    "agnews": 0.683,
    "dblp": 0.659,
    "books": 0.660,
    "reuters90": 0.637,
}
# V5 Y4 fractions (matched to the IS retention to 4 dp).
V5_Y4_FRACTION = {"agnews": 0.6831, "books": 0.6596, "reuters90": 0.637}

Y2_ARMS = ["raw_ep2", "raw_ep4"]
Y3_ARMS = ["sel_composite", "sel_entropy_only", "sel_margin_only", "sel_random_phases"]
Y4_ARM_BY_DATASET = {
    "agnews": "raw_rand0.6831",
    "books": "raw_rand0.6596",
    "reuters90": "raw_rand0.637",
}
Y5_ARMS = [f"sweep_b{b}_t{t}" for b in ("0.1", "0.3", "0.5") for t in ("0.0", "0.2", "0.4")]

# DC1-I2 declared factors: the only config keys allowed to differ between the
# arms of a campaign (compared within one dataset+fold).
DECLARED_FACTORS = {
    "Y1": {"mode"},
    "Y2": {"epochs"},
    "Y3": {"curriculum_method", "curriculum_margin_weight", "curriculum_entropy_weight"},
    "Y4": set(),
    "Y5": {"beta", "theta"},
}
# Bookkeeping / derived keys that are not configuration factors.
BOOKKEEPING = {"mode", "run_id", "timestamp", "experiment_id", "is_stats", "git_commit"}

# DC1-I2 named fields that the plan lists explicitly: asserted equal across the
# arms of a (campaign, dataset, fold) whenever they are not the declared factor.
DC1_I2_FIELDS = [
    "hf_model", "batch_size", "lr", "weight_decay", "warmup_ratio", "max_length",
    "random_state", "curriculum_q", "curriculum_phase_max_lengths",
    "curriculum_margin_weight", "curriculum_entropy_weight", "curriculum_length_weight",
]
# Config fields exported into the index.
CONFIG_COLUMNS = [
    "hf_model", "model", "max_length", "batch_size", "eval_batch_size", "lr",
    "weight_decay", "warmup_ratio", "random_state", "curriculum_method",
    "curriculum_q", "curriculum_margin_weight", "curriculum_entropy_weight",
    "curriculum_length_weight", "curriculum_phase_max_lengths",
    "epochs", "epochs_per_phase", "train_fraction", "beta", "theta",
]
TIMING_COLUMNS = [
    "data_load_time_s", "preprocess_time_s", "is_fit_time_s", "cl_signal_extract",
    "cl_phase_build", "train_subsample_time_s", "model_train_time_s",
    "metric_eval_time_s", "total_run_time_s",
]

_DIR_RE = re.compile(r"^(?P<dataset>.+?)-5cv-(?P<ts>\d{8}-\d{6})_(?P<suffix>.+)$")


def campaign_of(suffix: str) -> str | None:
    if suffix == "factorial_matched":
        return "Y1"
    if re.fullmatch(r"raw_ep[24]", suffix):
        return "Y2"
    if suffix.startswith("sel_"):
        return "Y3"
    if suffix.startswith("raw_rand"):
        return "Y4"
    if suffix.startswith("sweep_"):
        return "Y5"
    return None


def expected_cells() -> set[tuple[str, str, str, int]]:
    cells: set[tuple[str, str, str, int]] = set()
    for ds in DATASET_ORDER:
        for arm in ("cl", "is", "is_cl", "raw"):
            for f in range(5):
                cells.add(("Y1", ds, arm, f))
    for ds in ("agnews", "books", "reuters90"):
        for arm in Y2_ARMS:
            for f in range(5):
                cells.add(("Y2", ds, arm, f))
        for arm in Y3_ARMS:
            for f in range(5):
                cells.add(("Y3", ds, arm, f))
        for f in range(5):
            cells.add(("Y4", ds, Y4_ARM_BY_DATASET[ds], f))
    for ds in ("books", "reuters90"):
        for arm in Y5_ARMS:
            for f in range(3):
                cells.add(("Y5", ds, arm, f))
    return cells


# --------------------------------------------------------------------------- #
# Shard discovery
# --------------------------------------------------------------------------- #


def discover_shards(results_dir: Path):
    """Yield candidate arm shards.

    A shard = ``<dataset>-5cv-<ts>_<suffix>/<mode>_fold<k>/`` holding one
    (campaign, dataset, arm, fold) arm.  Returns (key, candidate-dict) pairs where
    the candidate records its timestamp and whether predictions are present.
    """
    candidates: dict[tuple, list[dict]] = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = _DIR_RE.match(d.name)
        if not m:
            continue
        camp = campaign_of(m.group("suffix"))
        if camp is None:
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir() or not (sub / "config.json").exists():
                continue
            cfg = _read_json(sub / "config.json")
            ds = cfg["dataset"]
            arm = cfg["mode"] if camp == "Y1" else m.group("suffix")
            fold = int(cfg["fold"])
            has_pred = (sub / "predictions_test.csv").exists()
            rec = {
                "campaign": camp, "dataset": ds, "arm": arm, "fold": fold,
                "dir": str(sub.relative_to(results_dir.parent)),
                "dirname": d.name, "ts": m.group("ts"), "suffix": m.group("suffix"),
                "has_pred": has_pred,
            }
            candidates.setdefault((camp, ds, arm, fold), []).append(rec)
    return candidates


def resolve(candidates):
    """Newest-complete duplicate resolution (V3)."""
    resolved: dict[tuple, dict] = {}
    dropped: list[dict] = []
    for key, cands in candidates.items():
        complete = [c for c in cands if c["has_pred"]]
        for c in cands:
            if not c["has_pred"]:
                dropped.append({**c, "reason": "incomplete: no predictions_test.csv"})
        if not complete:
            dropped.append({"key": key, "reason": "no complete shard"})
            continue
        chosen = max(complete, key=lambda c: c["ts"])
        for c in complete:
            if c is not chosen:
                dropped.append({**c, "reason": f"superseded by {chosen['dirname']}"})
        resolved[key] = chosen
    return resolved, dropped


# --------------------------------------------------------------------------- #
# Row extraction
# --------------------------------------------------------------------------- #


def _last_phase(pm: pd.DataFrame) -> pd.Series:
    return pm.iloc[-1]


def extract_row(rec: dict) -> dict:
    base = Path(rec["dir"])
    cfg = _read_json(base / "config.json")
    pm = _read_csv(base / "phase_metrics.csv")
    timings = _read_timings(base / "timings.csv")
    last = _last_phase(pm)

    # --- F1: recomputed from predictions (never from summary.csv) -------------
    pred = _read_csv(base / "predictions_test.csv")
    y_true = pred["y_true"].to_numpy()
    y_pred = pred["y_pred"].to_numpy()
    f1_final = float(f1_score(y_true, y_pred, average="macro"))
    ent = pred["pred_entropy"].to_numpy()
    q = float(last["hard_slice_quantile"])
    thr = np.quantile(ent, q)
    mask = ent >= thr
    hard_f1 = float(f1_score(y_true[mask], y_pred[mask], average="macro"))

    f1_last = float(last["macro_f1"])
    f1_max = float(pm["macro_f1"].max())
    hard_f1_stored = float(last["hard_slice_macro_f1"])

    # --- exposure (V10) ------------------------------------------------------
    tokens = float(last["compute_proxy"])
    avg_seq_len = float(last["avg_seq_len"])
    visits = tokens / avg_seq_len if avg_seq_len else float("nan")
    mult = cfg["epochs"] if len(pm) == 1 else cfg["epochs_per_phase"]
    visits_from_sum = float((pm["n_train_instances"].astype(float) * mult).sum())

    # --- cost decomposition (I8 / S4 rule) -----------------------------------
    nonfinal_pred = float(pm["pred_time_s"].iloc[:-1].sum())
    cost_load = timings.get("data_load_time_s", 0.0)
    cost_probe_sel = (
        timings.get("preprocess_time_s", 0.0)
        + timings.get("cl_signal_extract", 0.0)
        + timings.get("cl_phase_build", 0.0)
    )
    finetune = timings.get("model_train_time_s", 0.0) - nonfinal_pred
    total = timings.get("total_run_time_s", 0.0)
    cost_eval = total - cost_load - cost_probe_sel - finetune

    # --- selection ----------------------------------------------------------
    sel_path = base / "instance_selection.json"
    has_sel = sel_path.exists()
    if has_sel:
        sel = _read_json(sel_path)
        n_before = int(sel["n_train_before"])
        n_after = int(sel["n_train_after"])
        n_removed = int(sel["n_removed"])
        retention = n_after / n_before if n_before else float("nan")
        removed_by_class = json.dumps(sel.get("removed_by_class", {}), sort_keys=True)
        total_by_class = json.dumps(sel.get("total_by_class", {}), sort_keys=True)
    else:
        n_before = n_after = n_removed = retention = float("nan")
        removed_by_class = total_by_class = ""

    row = {
        "campaign": rec["campaign"], "dataset": rec["dataset"], "arm": rec["arm"],
        "fold": rec["fold"], "dir": rec["dir"], "commit": cfg.get("git_commit"),
        "n_phases": int(len(pm)),
        "f1_final": f1_final, "f1_last_phase": f1_last, "f1_max_phase": f1_max,
        "hard_f1": hard_f1, "hard_f1_stored": hard_f1_stored,
        "tokens": tokens, "visits": visits, "visits_from_sum": visits_from_sum,
        "avg_seq_len": avg_seq_len,
        "n_train": float(last["n_train_instances"]),
        "n_train_before": n_before, "n_train_after": n_after, "n_removed": n_removed,
        "retention": retention, "has_selection": has_sel,
        "removed_by_class": removed_by_class, "total_by_class": total_by_class,
        "cost_load": cost_load, "cost_probe_sel": cost_probe_sel,
        "cost_finetune": finetune, "cost_eval": cost_eval,
        # keep the raw logged stages for S4/debugging
        "nonfinal_pred_time_s": nonfinal_pred,
    }
    for key in CONFIG_COLUMNS:
        val = cfg.get(key)
        if isinstance(val, (list, tuple)):
            val = json.dumps(list(val))
        row[key] = val
    for key in TIMING_COLUMNS:
        row[key] = timings.get(key, float("nan"))
    # internal helpers used by the invariants
    row["_pred_idx"] = tuple(int(v) for v in pred["idx"].to_numpy())
    row["_pred_y_true"] = tuple(int(v) for v in y_true)
    row["_removed_dict"] = (
        json.loads(removed_by_class) if removed_by_class else None
    )
    row["_total_dict"] = json.loads(total_by_class) if total_by_class else None
    row["_cfg"] = cfg
    return row


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


class Log:
    def __init__(self):
        self.lines: list[str] = []
        self.failed = False

    def check(self, name: str, ok: bool, detail: str) -> bool:
        tag = "PASS" if ok else "FAIL"
        self.lines.append(f"[{tag}] {name}: {detail}")
        if not ok:
            self.failed = True
        return ok

    def info(self, name: str, detail: str) -> None:
        self.lines.append(f"[INFO] {name}: {detail}")

    def dump(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def run_invariants(rows: list[dict], dropped: list[dict], results_dir: Path, log: Log) -> None:
    df = pd.DataFrame(rows)
    by = {(r["campaign"], r["dataset"], r["arm"], r["fold"]): r for r in rows}

    # --- I1: cell completeness + fold counts --------------------------------
    observed = set(by)
    expected = expected_cells()
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    log.check(
        "I1a_cell_set", not missing and not extra,
        f"expected={len(expected)} observed={len(observed)} missing={len(missing)} extra={len(extra)}"
        + (f" missing_examples={missing[:5]}" if missing else "")
        + (f" extra_examples={extra[:5]}" if extra else ""),
    )
    bad_folds = []
    for (camp, ds, arm), g in df.groupby(["campaign", "dataset", "arm"]):
        want = {0, 1, 2} if camp == "Y5" else {0, 1, 2, 3, 4}
        got = set(int(f) for f in g["fold"])
        if got != want:
            bad_folds.append((camp, ds, arm, sorted(got)))
    log.check("I1b_fold_counts", not bad_folds,
              f"all (campaign,dataset,arm) fold sets OK; bad={bad_folds}")
    log.check("I1c_predictions_present", len(rows) == len(expected),
              f"{len(rows)} rows each with predictions_test.csv (complete shard)")

    # --- I2: config matching with only declared factors ----------------------
    named_offenders = []
    key_offenders = []
    for (camp, ds, fold), g in df.groupby(["campaign", "dataset", "fold"]):
        declared = DECLARED_FACTORS[camp]
        cfgs = list(g["_cfg"])
        keys = set().union(*[set(c) for c in cfgs])
        for key in sorted(keys):
            if key in declared or key in BOOKKEEPING:
                continue
            vals = {repr(c.get(key)) for c in cfgs}
            if len(vals) > 1:
                key_offenders.append((camp, ds, fold, key, sorted(vals)))
        for key in DC1_I2_FIELDS:
            if key in declared:
                continue
            vals = {repr(c.get(key)) for c in cfgs}
            if len(vals) > 1:
                named_offenders.append((camp, ds, fold, key))
    log.check("I2a_named_fields_equal", not named_offenders,
              f"DC1-I2 fields equal across arms except declared factors; bad={named_offenders[:5]}")
    log.check("I2b_no_undeclared_diff", not key_offenders,
              f"no undeclared config key differs across arms; bad={key_offenders[:5]}")

    # --- I3: V4 (max-phase == last-phase; recomputed == last-phase) ----------
    multi = df[df["n_phases"] > 1]
    d_max = float((multi["f1_max_phase"] - multi["f1_last_phase"]).abs().max()) if len(multi) else 0.0
    d_rec = float((df["f1_final"] - df["f1_last_phase"]).abs().max())
    d_hard = float((df["hard_f1"] - df["hard_f1_stored"]).abs().max())
    log.check("I3a_multiphase_count", len(multi) == 110, f"multi-phase folds = {len(multi)} (expect 110)")
    log.check("I3b_max_eq_last", d_max <= 1e-9, f"max |max_phase-last_phase| = {d_max:.2e}")
    log.check("I3c_recomputed_eq_last", d_rec <= 1e-6, f"max |recomputed-last_phase| = {d_rec:.2e}")
    log.check("I3d_hard_f1_recomputed", d_hard <= 1e-9, f"max |hard_f1_recomputed-last_phase_hard| = {d_hard:.2e}")

    # --- I4: paired test idx / y_true identical across arms of a fold --------
    pair_bad = []
    for (camp, ds, fold), g in df.groupby(["campaign", "dataset", "fold"]):
        idxs = {r["_pred_idx"] for _, r in g.iterrows()}
        ys = {r["_pred_y_true"] for _, r in g.iterrows()}
        if len(idxs) > 1 or len(ys) > 1:
            pair_bad.append((camp, ds, fold))
    log.check("I4_paired_idx_ytrue", not pair_bad, f"idx/y_true identical across arms; bad={pair_bad}")

    # --- I5: V6 frozen selections (Y3 all arms; Y5 (0.3,0.2)) --------------
    mismatch = []
    checked = 0
    for ds in ("agnews", "books", "reuters90"):
        ref: dict[int, dict] = {}
        for (camp, d2, arm, fold), r in by.items():
            if camp == "Y1" and d2 == ds and arm == "is":
                ref[fold] = r["_removed_dict"]
        for (camp, d2, arm, fold), r in by.items():
            if d2 != ds or camp not in ("Y3", "Y5"):
                continue
            if camp == "Y5" and arm != "sweep_b0.3_t0.2":
                continue
            checked += 1
            if r["_removed_dict"] != ref.get(fold):
                mismatch.append((camp, ds, arm, fold))
    log.check("I5_frozen_selection", not mismatch,
              f"Y3(all arms)+Y5(0.3,0.2) removed_by_class == Y1 IS; checked={checked} bad={mismatch}")

    # --- I6: V5 retention + Y4 fractions (recomputed) -----------------------
    ret_off = {}
    for ds, expect in V5_RETENTION.items():
        sub = df[(df["campaign"] == "Y1") & (df["dataset"] == ds) & (df["arm"] == "is")]
        got = float(sub["retention"].mean())
        if abs(got - expect) > 5e-4:
            ret_off[ds] = (round(got, 6), expect)
    log.check("I6a_v5_retention", not ret_off, f"Y1 IS fold-mean retention matches V5; bad={ret_off}")

    y4_off = {}
    y4_ndiff = {}
    for ds, y4arm in Y4_ARM_BY_DATASET.items():
        y1raw = {(r["fold"]): r["n_train"] for r in rows
                 if r["campaign"] == "Y1" and r["dataset"] == ds and r["arm"] == "raw"}
        y4 = {(r["fold"]): r["n_train"] for r in rows
              if r["campaign"] == "Y4" and r["dataset"] == ds}
        ratios = [y4[f] / y1raw[f] for f in sorted(y4) if f in y1raw]
        frac = sum(ratios) / len(ratios)
        isret = float(df[(df["campaign"] == "Y1") & (df["dataset"] == ds)
                         & (df["arm"] == "is")]["retention"].mean())
        if round(frac, 4) != round(isret, 4):
            y4_off[ds] = (round(frac, 6), round(isret, 6))
        y4_ndiff[ds] = max(abs(y4[f] - y1raw[f] * frac) for f in sorted(y4) if f in y1raw)
    log.check("I6b_v5_y4_fraction", not y4_off,
              f"Y4 subsample fraction == IS retention to 4 dp; bad={y4_off}; max|dn|={y4_ndiff}")

    # --- I7: V10 visit derivations agree ------------------------------------
    dv = float((df["visits"] - df["visits_from_sum"]).abs().max())
    log.check("I7_visits_agree", dv <= 1.0,
              f"max |compute_proxy/avg_seq_len - sum(n*epochs)| = {dv:.4f}")

    # --- I8: nonnegative additive costs -------------------------------------
    stages = [c for c in TIMING_COLUMNS]
    neg = {c: int((df[c] < -1e-9).sum()) for c in stages if (df[c] < -1e-9).any()}
    comp_neg = {c: int((df[c] < -1e-6).sum())
                for c in ("cost_load", "cost_probe_sel", "cost_finetune", "cost_eval")
                if (df[c] < -1e-6).any()}
    resid = (df["cost_load"] + df["cost_probe_sel"] + df["cost_finetune"] + df["cost_eval"]
             - df["total_run_time_s"]).abs().max()
    log.check("I8a_stages_nonneg", not neg, f"logged stages >= 0; negatives={neg}")
    log.check("I8b_components_nonneg", not comp_neg, f"cost components >= 0; negatives={comp_neg}")
    log.check("I8c_sum_to_total", float(resid) <= 1e-6,
              f"max |sum(components)-total| = {float(resid):.2e}")

    # --- I9: source-commit equality -----------------------------------------
    commits = sorted(set(df["commit"].dropna()))
    camp_commits = {c: sorted(set(g["commit"])) for c, g in df.groupby("campaign")}
    try:
        diff = subprocess.run(
            ["git", "diff", "--stat", "4bed017", "61bc115", "--", "src"],
            capture_output=True, text=True, cwd=str(results_dir.parent), check=True,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover
        diff = f"__ERROR__ {exc}"
    log.check("I9a_src_diff_empty", diff == "",
              f"git diff 4bed017..61bc115 -- src is empty (got {len(diff)} chars)")
    ok_commits = (
        camp_commits.get("Y1") == ["4bed017"]
        and all(camp_commits.get(c) == ["61bc115"] for c in ("Y2", "Y3", "Y4", "Y5"))
    )
    log.check("I9b_commit_per_campaign", ok_commits, f"commits per campaign = {camp_commits}; all={commits}")

    # --- I10: dataset order (V8) --------------------------------------------
    y1 = df[df["campaign"] == "Y1"]
    size = y1.groupby("dataset")["n_train"].mean().sort_values(ascending=False)
    order = list(size.index)
    log.check("I10_dataset_order", order == DATASET_ORDER,
              f"descending training size order = {order} (V8 expects {DATASET_ORDER})")

    # --- I11: no torch / transformers ---------------------------------------
    leaked = sorted({"torch", "transformers"} & set(sys.modules))
    log.check("I11_no_torch_transformers", not leaked, f"leaked modules = {leaked}")

    # --- informational: V3 duplicate + V14 total GPU-hours -------------------
    for d in dropped:
        log.info("V3_duplicate_dropped",
                 f"{d.get('dir', d.get('key'))} ({d['reason']})")
    total_gpu_h = float(df["total_run_time_s"].sum()) / 3600.0
    log.info("V14_total_gpu_hours", f"deduplicated sum of total_run_time_s = {total_gpu_h:.2f} GPU-h")
    raw_dirs = [d for d in results_dir.iterdir()]
    log.info("shards_scanned", f"{len(raw_dirs)} top-level dirs; {len(rows)} resolved rows")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the SPURS matched-runs index (S1).")
    ap.add_argument("--results-dir", default="runpod_results")
    ap.add_argument("--out-dir", default="analysis/paper/out")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = discover_shards(results_dir)
    resolved, dropped = resolve(candidates)
    rows = [extract_row(rec) for _, rec in sorted(resolved.items())]

    # retention for rows without an instance_selection.json (raw arms): the Y4
    # random-stratified arms subsample the fold to a measured fraction, so
    # retention = n_train / n_train(Y1 raw, same dataset+fold); other raw rows
    # keep the full training fold -> 1.0.
    y1raw_n = {
        (r["dataset"], r["fold"]): r["n_train"]
        for r in rows if r["campaign"] == "Y1" and r["arm"] == "raw"
    }
    for r in rows:
        if r["has_selection"]:
            r["retention_effective"] = r["retention"]
        else:
            tf = r.get("train_fraction")
            ref = y1raw_n.get((r["dataset"], r["fold"]))
            if tf is not None and float(tf) < 1.0 and ref:
                r["retention_effective"] = r["n_train"] / ref
            else:
                r["retention_effective"] = 1.0

    # stable ordering: campaign, dataset order, arm, fold
    ds_rank = {d: i for i, d in enumerate(DATASET_ORDER)}
    rows.sort(key=lambda r: (r["campaign"], ds_rank.get(r["dataset"], 99), r["arm"], r["fold"]))

    df = pd.DataFrame(rows)
    df["dataset_order"] = df["dataset"].map(ds_rank)
    # drop internal helper columns before writing
    cols = [c for c in df.columns if not c.startswith("_")]
    lead = ["campaign", "dataset", "dataset_order", "arm", "fold", "dir", "commit"]
    ordered = lead + ["n_phases", "f1_final", "f1_last_phase", "f1_max_phase", "hard_f1"] \
        + CONFIG_COLUMNS + [c for c in TIMING_COLUMNS] \
        + ["cost_load", "cost_probe_sel", "cost_finetune", "cost_eval", "nonfinal_pred_time_s"] \
        + ["tokens", "visits", "visits_from_sum", "avg_seq_len", "n_train",
           "n_train_before", "n_train_after", "n_removed", "retention",
           "retention_effective", "has_selection",
           "removed_by_class", "total_by_class"]
    ordered = [c for c in ordered if c in cols] + [c for c in cols if c not in ordered]
    out_csv = out_dir / "runs_index.csv"
    df[ordered].to_csv(out_csv, index=False)

    log = Log()
    log.info("index_written", f"{out_csv} rows={len(df)} cols={len(ordered)}")
    log.info("results_dir", str(results_dir))
    run_invariants(rows, dropped, results_dir, log)
    log.dump(out_dir / "invariants.log")

    # guard against reading summary.csv anywhere
    assert not any(p.endswith("summary.csv") for p in OPENED_FILES), "summary.csv was read"
    for mod in ("torch", "transformers"):
        assert mod not in sys.modules, f"{mod} leaked into sys.modules"

    print(f"wrote {out_csv} ({len(df)} rows) and {out_dir / 'invariants.log'}")
    for line in log.lines:
        print(line)
    return 1 if log.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
