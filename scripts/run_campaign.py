#!/usr/bin/env python3
"""RunPod campaign runner — resumable, ledger-backed, one shard at a time.

WHY THIS EXISTS
`cli/experiment.py` has no resume support: every invocation with `timestamp: auto`
creates a fresh results directory, and nothing skips work already done. On an
interruptible RunPod pod, relaunching by hand either duplicates work or loses it.
This runner wraps the CLI in a ledger so a campaign survives preemption.

WHY IT DOES NOT IMPORT bio_is_curriculum
Importing `bio_is_curriculum.config.cuda` calls `configure_cuda_device()` at
import time, which would repin `CUDA_VISIBLE_DEVICES` in the runner's own process
and leak that into every child. The runner therefore shells out to the CLI and
never imports the package. `yaml` is fine: it is a declared project dependency
(pyproject.toml), not the package under test — and a hand-rolled parser for
nested flow mappings would be the fragile choice, not the safe one.

SHARD GRANULARITY
The CLI has no `--mode` flag, so the finest slice it accepts is (dataset, fold):
each shard runs every mode of that dataset-fold (4 in Y1: raw/is/cl/is_cl). One
shard per fold bounds the work lost to a preemption to a single fold — ~13 h for
a yelp_2013 fold, versus 65 h if shards were whole datasets.

COMPLETION IS THREE CHECKS, NOT ONE
A results directory is the worst completion signal: a killed run leaves a partial
`config.json`, and because `experiment_id` carries a fresh timestamp each
invocation, mere existence cannot tell partial from complete. A manifest is
better but is also written for batches that FAILED (it precedes `sys.exit(1)`),
so presence is not success either. A shard counts as done only when all three
hold:

  1. a marker file in state/done/ (written atomically via os.replace), AND
  2. the manifest it points at has every run with `status == "ok"`, AND
  3. every expected artifact exists under each run directory.

The only failure mode is "succeeded, then died before writing the marker", which
re-runs the shard — the safe direction.

USAGE
  uv run python scripts/run_campaign.py --dry-run          # list the shards
  uv run python scripts/run_campaign.py                    # run them all
  uv run python scripts/run_campaign.py --only reuters90:0 # one shard
  uv run python scripts/run_campaign.py --status           # progress + heartbeat
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# /workspace is the RunPod volume. Off-pod (for --dry-run / --status) fall back
# to a user cache directory — deliberately outside the repo, so a local
# invocation cannot leave state behind in the working tree.
_DEFAULT_WORKSPACE = Path("/workspace")
WORKSPACE = Path(os.environ.get("WORKSPACE", str(_DEFAULT_WORKSPACE)))
if not WORKSPACE.is_dir():
    WORKSPACE = Path.home() / ".cache" / "bio-is-curriculum"

STATE_DIR = Path(os.environ.get("BIO_STATE_DIR", str(WORKSPACE / "state")))
LOG_DIR = Path(os.environ.get("BIO_LOG_DIR", str(WORKSPACE / "logs")))
LEDGER = STATE_DIR / "ledger.jsonl"
DONE_DIR = STATE_DIR / "done"
HEARTBEAT = STATE_DIR / "heartbeat.json"

CAMPAIGN_DIR = REPO_ROOT / "experiments" / "campaigns"

# Shipped campaigns, by short name.
CAMPAIGNS: dict[str, str] = {
    "y1": "factorial_matched_2x2.yaml",
    "y2": "raw_curve_diagnostic.yaml",
    "y3": "controls_signal_same_selection.yaml",
    "y4": "controls_random_stratified.yaml",
    "y5": "beta_theta_sweep.yaml",
}

# A campaign carrying this marker has an unfilled measured value; the runner
# refuses to execute it. See controls_random_stratified.yaml.
PLACEHOLDER_MARKER = "TODO(placeholder)"

# Artifacts every completed run directory must contain. predictions_test.csv is
# the substrate for bio-equivalence / per-class breakdowns, and runner.py calls
# save_predictions() unconditionally on both the curriculum and the plain path.
REQUIRED_ARTIFACTS = (
    "config.json",
    "phase_metrics.csv",
    "timings.csv",
    "predictions_test.csv",
)

# GPU-hours per (dataset, fold) for the Y1 2² — 4 modes per shard. Sources: the
# header of factorial_matched_2x2.yaml and docs/experimentos_maquina.md §5
# (reuters90 2.7 / books 6.9 / dblp 7.7 / agnews 12.3 / yelp_2013 65.1 GPU-h over
# 5 folds; medline 151.1). Used only for shard ordering and the calibration gate.
HOURS_PER_FOLD_4_MODES: dict[str, float] = {
    "reuters90": 2.7 / 5,
    "books": 6.9 / 5,
    "dblp": 7.7 / 5,
    "agnews": 12.3 / 5,
    "yelp_2013": 65.1 / 5,
    "medline": 151.1 / 5,
}

# 130 = SIGINT, 137 = SIGKILL (preemption or the OOM killer), 143 = SIGTERM.
PREEMPTION_CODES = frozenset({130, 137, 143})

EXPERIMENTS_SUBDIR = "experiments"
ORPHAN_DIRNAME = "_orphans"

HEARTBEAT_INTERVAL_S = 60.0
LOG_TAIL_BYTES = 8192


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def shard_key(dataset: str, fold: int) -> str:
    return f"{dataset}_f{fold}"


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def append_ledger(record: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def read_ledger() -> list[dict]:
    if not LEDGER.is_file():
        return []
    records: list[dict] = []
    with LEDGER.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A line torn by a hard kill. Skip it; the marker file is the
                # authoritative completion signal, not the ledger.
                continue
    return records


def tail_of(path: Path, nbytes: int = LOG_TAIL_BYTES) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as fh:
        try:
            fh.seek(-nbytes, os.SEEK_END)
        except OSError:
            fh.seek(0)
        return fh.read().decode("utf-8", errors="replace")


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def gpu_name() -> str:
    """Best-effort GPU label for provenance. Cheap, and no torch import."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return ", ".join(names) if names else "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


# --------------------------------------------------------------------------- #
# Shard model
# --------------------------------------------------------------------------- #
@dataclass
class Shard:
    dataset: str
    fold: int
    declared_hours: float

    @property
    def key(self) -> str:
        return shard_key(self.dataset, self.fold)

    @property
    def label(self) -> str:
        return f"{self.dataset}/fold{self.fold}"


@dataclass
class Campaign:
    name: str
    path: Path
    text: str
    doc: dict
    results_dir: str
    shards: list[Shard] = field(default_factory=list)
    modes_per_shard: int = 4

    @property
    def experiments_root(self) -> Path:
        return REPO_ROOT / self.results_dir / EXPERIMENTS_SUBDIR

    @property
    def results_root(self) -> Path:
        return REPO_ROOT / self.results_dir


def load_campaign(short_name: str) -> Campaign:
    if short_name not in CAMPAIGNS:
        raise SystemExit(
            f"Unknown campaign {short_name!r}. Known: {', '.join(sorted(CAMPAIGNS))}"
        )
    path = CAMPAIGN_DIR / CAMPAIGNS[short_name]
    text = path.read_text(encoding="utf-8")
    doc = yaml.safe_load(text) or {}
    campaign_doc = doc.get("campaign") or {}

    defaults = campaign_doc.get("defaults") or {}
    results_dir = str(defaults.get("results_dir", doc.get("results_dir", "results")))

    # Shards are (dataset, fold). n_splits per dataset may override the default;
    # folds come from defaults.folds when set, else 0..n_splits-1.
    default_n_splits = int(defaults.get("n_splits", 10))
    default_folds = defaults.get("folds")

    shards: list[Shard] = []
    for dataset, ds_cfg in (campaign_doc.get("datasets") or {}).items():
        ds_cfg = ds_cfg or {}
        n_splits = int(ds_cfg.get("n_splits", default_n_splits))
        folds = ds_cfg.get("folds", default_folds)
        fold_ids = [int(f) for f in folds] if folds else list(range(n_splits))
        hours = HOURS_PER_FOLD_4_MODES.get(dataset, float("nan"))
        for fold in fold_ids:
            shards.append(Shard(dataset=dataset, fold=fold, declared_hours=hours))

    # Modes of the first job, so the calibration budget can be scaled for
    # campaigns that do not run all four modes.
    modes_per_shard = 4
    jobs = campaign_doc.get("jobs") or []
    if jobs and jobs[0].get("modes"):
        modes_per_shard = max(1, len(jobs[0]["modes"]))

    return Campaign(
        name=str(campaign_doc.get("name") or path.stem),
        path=path,
        text=text,
        doc=doc,
        results_dir=results_dir,
        shards=shards,
        modes_per_shard=modes_per_shard,
    )


def order_shards(shards: list[Shard], order: str) -> list[Shard]:
    """Cheap-first by default, so the calibration shard is the first one run.

    Datasets absent from HOURS_PER_FOLD_4_MODES (nan) sort last: an unknown cost
    is not evidence of being cheap.
    """
    def sort_key(shard: Shard):
        hours = shard.declared_hours
        unknown = hours != hours  # NaN
        return (1 if unknown else 0, 0.0 if unknown else hours, shard.dataset, shard.fold)

    ordered = sorted(shards, key=sort_key)
    if order == "big-first":
        known = [s for s in ordered if s.declared_hours == s.declared_hours]
        unknown = [s for s in ordered if s.declared_hours != s.declared_hours]
        ordered = list(reversed(known)) + unknown
    return ordered


def filter_shards(shards: list[Shard], only: list[str] | None) -> list[Shard]:
    if not only:
        return shards
    wanted: set[tuple[str, int | None]] = set()
    for spec in only:
        if ":" in spec:
            dataset, _, fold = spec.partition(":")
            wanted.add((dataset, int(fold)))
        else:
            wanted.add((spec, None))
    kept = [
        s for s in shards
        if (s.dataset, s.fold) in wanted or (s.dataset, None) in wanted
    ]
    unknown = {d for d, _ in wanted} - {s.dataset for s in shards}
    if unknown:
        raise SystemExit(f"--only names unknown dataset(s): {', '.join(sorted(unknown))}")
    return kept


# --------------------------------------------------------------------------- #
# Completion verification
# --------------------------------------------------------------------------- #
def manifest_runs_ok(manifest: dict) -> tuple[bool, str]:
    runs = manifest.get("runs") or []
    if not runs:
        return False, "manifest has no runs"
    bad = [r.get("experiment_id") for r in runs if r.get("status") != "ok"]
    if bad:
        return False, f"{len(bad)} run(s) not ok: {', '.join(str(b) for b in bad[:3])}"
    return True, ""


def verify_shard(campaign: Campaign, shard: Shard, marker: dict) -> tuple[bool, str]:
    """Full completion check: manifest ok AND every artifact present."""
    manifest_path = Path(str(marker.get("manifest_path", "")))
    if not manifest_path.is_file():
        return False, f"manifest missing: {manifest_path}"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"manifest unreadable: {exc}"

    ok, why = manifest_runs_ok(manifest)
    if not ok:
        return False, why

    for run in manifest.get("runs") or []:
        run_path = Path(str(run.get("path", "")))
        if not run_path.is_absolute():
            run_path = REPO_ROOT / run_path
        for mode in run.get("modes") or []:
            run_dir = run_path / f"{mode}_fold{shard.fold}"
            missing = [a for a in REQUIRED_ARTIFACTS if not (run_dir / a).is_file()]
            if missing:
                return False, f"{run_dir.name}: missing {', '.join(missing)}"
    return True, ""


def is_complete(campaign: Campaign, shard: Shard) -> tuple[bool, str]:
    marker_path = DONE_DIR / f"{shard.key}.json"
    if not marker_path.is_file():
        return False, "no marker"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"marker unreadable: {exc}"
    return verify_shard(campaign, shard, marker)


# --------------------------------------------------------------------------- #
# Executing a shard
# --------------------------------------------------------------------------- #
def shard_command(campaign: Campaign, shard: Shard) -> list[str]:
    """The shard's command.

    sys.executable, not `uv run bio-experiment`: the runner is already running in
    the project venv, and a nested `uv run` per shard would re-resolve the
    environment each time. `-m bio_is_curriculum.cli.experiment` is the same entry
    point the console script calls.

    --no-docker is mandatory: pods have no Docker daemon, and experiment.py
    selects Docker whenever the YAML has a `docker:` block (all 24 do).
    cwd must be REPO_ROOT because results_dir and data_dir are relative.
    """
    return [
        sys.executable, "-m", "bio_is_curriculum.cli.experiment",
        str(campaign.path.relative_to(REPO_ROOT)),
        "--no-docker",
        "--dataset", shard.dataset,
        "--folds", str(shard.fold),
    ]


def snapshot(results_root: Path) -> set[Path]:
    """Top-level entries of results/ and of results/experiments/."""
    seen: set[Path] = set()
    for root in (results_root, results_root / EXPERIMENTS_SUBDIR):
        if root.is_dir():
            seen.update(root.iterdir())
    return seen


def orphan_new_dirs(results_root: Path, before: set[Path]) -> list[Path]:
    """Move everything a failed attempt created out of the way.

    Partial directories carry distinct timestamps and would otherwise pollute
    `bio-summary` and every discovery pass that scans results/.
    """
    after = snapshot(results_root)
    orphans: list[Path] = []
    orphan_root = results_root / ORPHAN_DIRNAME
    for path in sorted(after - before):
        if path.name == ORPHAN_DIRNAME:
            continue
        orphan_root.mkdir(parents=True, exist_ok=True)
        target = orphan_root / path.name
        suffix = 1
        while target.exists():
            target = orphan_root / f"{path.name}.{suffix}"
            suffix += 1
        try:
            shutil.move(str(path), str(target))
            orphans.append(target)
        except OSError as exc:
            print(f"  warn: could not orphan {path}: {exc}")
    return orphans


def write_heartbeat(shard: Shard, attempt: int, pid: int | None) -> None:
    atomic_write_json(
        HEARTBEAT,
        {
            "shard": shard.key,
            "label": shard.label,
            "attempt": attempt,
            "pid": pid,
            "updated_at": now_iso(),
        },
    )


def classify(returncode: int, tail: str) -> tuple[str, str]:
    """Map an exit status to (kind, detail).

    kind is one of: ok, preempted, oom, poison. Preemption is environmental and
    retried at once; poison is deterministic and never retried.
    """
    if returncode == 0:
        return "ok", ""
    if returncode < 0:
        return "preempted", f"killed by signal {-returncode}"
    if returncode in PREEMPTION_CODES:
        return "preempted", f"exit {returncode}"
    if "CUDA out of memory" in tail or "OutOfMemoryError" in tail:
        return "oom", "CUDA out of memory"
    if "Traceback (most recent call last)" in tail:
        return "poison", "python traceback"
    return "poison", f"exit {returncode}"


def run_attempt(campaign: Campaign, shard: Shard, attempt: int) -> tuple[int, Path]:
    """Run one attempt, streaming output to a per-attempt log.

    Popen + poll rather than subprocess.run so the heartbeat keeps ticking and a
    Ctrl-C tears down the child instead of orphaning a training process.
    """
    log_path = LOG_DIR / campaign.name / f"{shard.key}__attempt{attempt}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = shard_command(campaign, shard)
    print(f"  $ {' '.join(cmd)}")
    print(f"  log: {log_path}")

    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"# {' '.join(cmd)}\n# started {now_iso()}\n")
        log_fh.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), env=env,
            stdout=log_fh, stderr=subprocess.STDOUT, text=True,
        )
        write_heartbeat(shard, attempt, proc.pid)
        last_beat = time.monotonic()
        try:
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                now = time.monotonic()
                if now - last_beat >= HEARTBEAT_INTERVAL_S:
                    write_heartbeat(shard, attempt, proc.pid)
                    last_beat = now
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n  interrupted — terminating child ...")
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    return proc.returncode, log_path


def mark_done(campaign: Campaign, shard: Shard, record: dict) -> None:
    atomic_write_json(DONE_DIR / f"{shard.key}.json", record)


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
def print_status(campaign: Campaign, shards: list[Shard]) -> int:
    records = read_ledger()
    by_status: dict[str, int] = {}
    for rec in records:
        by_status[rec.get("status", "?")] = by_status.get(rec.get("status", "?"), 0) + 1

    print(f"Campaign : {campaign.name} ({campaign.path.name})")
    print(f"State    : {STATE_DIR}")
    if records:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
        print(f"Ledger   : {len(records)} attempt(s) — {summary}")
    else:
        print(f"Ledger   : empty ({LEDGER})")

    done = 0
    pending: list[Shard] = []
    for shard in shards:
        complete, _ = is_complete(campaign, shard)
        if complete:
            done += 1
        else:
            pending.append(shard)
    print(f"Shards   : {done}/{len(shards)} complete, {len(pending)} pending")
    if pending:
        print(f"  next   : {', '.join(s.label for s in pending[:8])}"
              + (" ..." if len(pending) > 8 else ""))

    if HEARTBEAT.is_file():
        try:
            hb = json.loads(HEARTBEAT.read_text(encoding="utf-8"))
            age = datetime.now(timezone.utc) - datetime.fromisoformat(hb["updated_at"])
            mins = age.total_seconds() / 60.0
            print(f"Heartbeat: {hb.get('label')} attempt {hb.get('attempt')} "
                  f"pid {hb.get('pid')} — {mins:.1f} min ago")
            # A stalled heartbeat with a shard marked running is the signature of
            # a silent CPU fallback: it looks like a slow run, but it is alive.
            if mins > 10 and hb.get("shard") in {s.key for s in pending}:
                print("  WARNING: heartbeat is stale while a shard is unfinished. "
                      "Check whether the run fell back to CPU (see pod_bootstrap.sh "
                      "step 9) or was preempted.")
            log = LOG_DIR / campaign.name / f"{hb.get('shard')}__attempt{hb.get('attempt')}.log"
            tail = tail_of(log, 2000).strip().splitlines()[-12:]
            if tail:
                print("  --- log tail ---")
                for line in tail:
                    print(f"  {line}")
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as exc:
            print(f"Heartbeat: unreadable ({exc})")
    else:
        print("Heartbeat: none")

    return 0


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def preflight(campaign: Campaign, *, executing: bool) -> None:
    """Guards that must hold before spending GPU-hours."""
    if not (REPO_ROOT / "src" / "bio_is_curriculum" / "cli" / "experiment.py").is_file():
        raise SystemExit(f"FAIL: {REPO_ROOT} does not look like the repo root")

    # The GPU trap. configure_cuda_device() skips pinning when running_in_docker()
    # is true; BIO_IS_IN_DOCKER=1 is what makes it true. Without it, a bare-metal
    # run pins physical GPU 7 (DEFAULTS["cuda_device_id"]) and a 1-GPU pod silently
    # trains on CPU — which looks exactly like a slow run. Refuse rather than warn.
    if executing and os.environ.get("BIO_IS_IN_DOCKER") != "1":
        raise SystemExit(
            "FAIL: BIO_IS_IN_DOCKER is not '1'.\n"
            "  Without it, config/cuda.py::configure_cuda_device() pins physical GPU 7\n"
            "  and a single-GPU pod falls back to CPU with no error.\n"
            "  Run scripts/pod_bootstrap.sh, or: source /workspace/env.sh"
        )
    if executing and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        print("  WARN: CUDA_VISIBLE_DEVICES is unset; BIO_IS_IN_DOCKER=1 still "
              "prevents pinning, but set it explicitly (see pod_bootstrap.sh).")

    if executing and PLACEHOLDER_MARKER in campaign.text:
        raise SystemExit(
            f"FAIL: {campaign.path.name} still contains {PLACEHOLDER_MARKER}.\n"
            "  Its train_fraction must be the MEASURED 1 - is_reduction from Y1's\n"
            "  summary, not a guess — a wrong denominator makes the control\n"
            "  meaningless. Fill it in and delete the marker."
        )
    if not executing and PLACEHOLDER_MARKER in campaign.text:
        print(f"  NOTE: {campaign.path.name} is a placeholder campaign "
              f"({PLACEHOLDER_MARKER}); it cannot be executed yet.")

    if not campaign.shards:
        raise SystemExit("FAIL: campaign expands to zero shards")


def calibration_gate(shard: Shard, elapsed_s: float, modes_per_shard: int,
                     *, no_gate: bool) -> None:
    """Abort if the first shard overruns its declared budget by 2x.

    The first shard is the cheapest one and doubles as the runbook's mandatory
    smoke calibration. If it is far off, the budget table is wrong and the rest of
    the campaign should be rescaled rather than discovered to be over-budget 20
    hours in.
    """
    if no_gate or shard.declared_hours != shard.declared_hours:  # NaN
        return
    budget_h = shard.declared_hours * (modes_per_shard / 4.0)
    elapsed_h = elapsed_s / 3600.0
    print(f"  calibration: {elapsed_h:.2f} h vs declared {budget_h:.2f} h")
    if elapsed_h > 2.0 * budget_h:
        raise SystemExit(
            f"FAIL: calibration shard {shard.label} took {elapsed_h:.2f} h, more than "
            f"2x the declared {budget_h:.2f} h.\n"
            "  The cost table in this script is off, or the pod is slower than the\n"
            "  A100 the paper used. Rescale the budget before continuing, or pass\n"
            "  --no-gate to proceed anyway."
        )


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run_campaign(campaign: Campaign, shards: list[Shard], args) -> int:
    poison: list[str] = []
    abandoned: list[str] = []
    completed = 0
    processed = 0
    first_shard = True

    for shard in shards:
        if args.stop_after is not None and processed >= args.stop_after:
            print(f"\n--stop-after {args.stop_after} reached; stopping.")
            break

        complete, why = is_complete(campaign, shard)
        if complete:
            print(f"[skip] {shard.label} — already complete")
            completed += 1
            continue

        processed += 1
        print(f"\n{'=' * 65}\n  {shard.label}   ({why})\n{'=' * 65}")

        attempt = 0
        oom_retries = 0
        last_kind = ""
        last_detail = ""
        while True:
            attempt += 1
            started = now_iso()
            t0 = time.perf_counter()
            before = snapshot(campaign.results_root)

            try:
                returncode, log_path = run_attempt(campaign, shard, attempt)
            except KeyboardInterrupt:
                orphan_new_dirs(campaign.results_root, before)
                print("Interrupted. Partial output moved to "
                      f"{campaign.results_root / ORPHAN_DIRNAME}/")
                return 130

            elapsed = time.perf_counter() - t0
            tail = tail_of(log_path)
            kind, detail = classify(returncode, tail)
            last_kind, last_detail = kind, detail

            record = {
                "shard": shard.key,
                "campaign": campaign.name,
                "dataset": shard.dataset,
                "fold": shard.fold,
                "attempt": attempt,
                "started_at": started,
                "finished_at": now_iso(),
                "duration_s": round(elapsed, 2),
                "exit_code": returncode,
                "status": kind,
                "detail": detail,
                "log_path": str(log_path),
                "manifest_path": None,
                "run_dirs": [],
                "gpu_name": args.gpu_name,
                "git_commit": args.git_commit,
                "hostname": socket.gethostname(),
                "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            }

            if kind == "ok":
                after = snapshot(campaign.results_root)
                new_experiment_dirs = sorted(
                    p for p in (after - before)
                    if p.parent == campaign.experiments_root
                )
                new_run_dirs = sorted(
                    p for p in (after - before)
                    if p.parent == campaign.results_root and p.name != EXPERIMENTS_SUBDIR
                )
                record["run_dirs"] = [str(p) for p in new_run_dirs]
                if len(new_experiment_dirs) != 1:
                    # Timestamp collision, or the CLI wrote somewhere unexpected.
                    # The manifest is the completion evidence, so refuse to guess.
                    kind = "poison"
                    detail = (f"expected exactly 1 new experiment dir, "
                              f"found {len(new_experiment_dirs)}")
                    record["status"], record["detail"] = kind, detail
                else:
                    manifest_path = new_experiment_dirs[0] / "manifest.json"
                    record["manifest_path"] = str(manifest_path)
                    marker_ok, marker_why = verify_shard(
                        campaign, shard, {"manifest_path": str(manifest_path)}
                    )
                    if marker_ok:
                        append_ledger(record)
                        mark_done(campaign, shard, record)
                        write_heartbeat(shard, attempt, None)
                        completed += 1
                        print(f"  OK in {elapsed / 60:.1f} min — marker written")
                        if first_shard:
                            calibration_gate(shard, elapsed, campaign.modes_per_shard,
                                             no_gate=args.no_gate)
                            first_shard = False
                        break
                    kind = "poison"
                    detail = f"exit 0 but incomplete: {marker_why}"
                    record["status"], record["detail"] = kind, detail

            # Failure path.
            append_ledger(record)
            print(f"  {kind.upper()}: {detail} ({elapsed / 60:.1f} min, attempt {attempt})")

            if kind == "poison":
                # Deterministic: retrying burns GPU-hours to reproduce the same
                # traceback. Report and move on; one bad shard must not kill the
                # campaign.
                poison.append(f"{shard.label}: {detail}")
                orphan_new_dirs(campaign.results_root, before)
                break

            if kind == "oom":
                oom_retries += 1
                if oom_retries > 1:
                    poison.append(f"{shard.label}: CUDA OOM twice")
                    orphan_new_dirs(campaign.results_root, before)
                    break

            if attempt > args.retry_cap:
                # Out of attempts. Distinguish "the pod kept killing us" from
                # "our code is broken" — they need different responses.
                if last_kind == "preempted":
                    abandoned.append(f"{shard.label}: {last_detail} "
                                     f"({attempt} attempts)")
                else:
                    poison.append(f"{shard.label}: {last_detail} "
                                  f"({attempt} attempts)")
                orphan_new_dirs(campaign.results_root, before)
                break

            print(f"  retrying {shard.label} (attempt {attempt + 1}) ...")
            orphan_new_dirs(campaign.results_root, before)

        if poison and args.fail_fast:
            print("\n--fail-fast: stopping after the first poison shard.")
            break

    print(f"\n{'=' * 65}")
    print(f"  {completed}/{len(shards)} shard(s) complete")
    if abandoned:
        print(f"  {len(abandoned)} abandoned (preemption):")
        for item in abandoned:
            print(f"    - {item}")
    if poison:
        print(f"  {len(poison)} poison shard(s):")
        for item in poison:
            print(f"    - {item}")
    print(f"{'=' * 65}")
    print("\nFinal pass (per shard manifest is recorded in the ledger):")
    print("  uv run bio-summary <experiment_dir>")
    print("  uv run bio-equivalence <experiment_dir>")
    print("  uv run bio-significance <experiment_dir>")

    return 1 if (poison or abandoned) else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resumable RunPod campaign runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--campaign", default="y1", choices=sorted(CAMPAIGNS),
                        help="Which campaign to run (default: y1).")
    parser.add_argument("--only", action="append", metavar="DS[:FOLD]",
                        help="Restrict to a dataset, or one dataset:fold. Repeatable.")
    parser.add_argument("--order", choices=["cheap-first", "big-first"],
                        default="cheap-first",
                        help="Shard order (default: cheap-first).")
    parser.add_argument("--retry-cap", type=int, default=2,
                        help="Retries after the first attempt (default: 2 -> 3 attempts).")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="Stop after N shards have been attempted.")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop at the first poison shard instead of continuing.")
    parser.add_argument("--no-gate", action="store_true",
                        help="Skip the calibration gate on the first shard.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the shards and their commands; run nothing.")
    parser.add_argument("--status", action="store_true",
                        help="Print ledger summary, heartbeat age and current log tail.")
    args = parser.parse_args()

    campaign = load_campaign(args.campaign)
    shards = order_shards(campaign.shards, args.order)
    shards = filter_shards(shards, args.only)

    if args.status:
        return print_status(campaign, shards)

    print(f"Campaign : {campaign.name} ({campaign.path.name})")
    print(f"Repo     : {REPO_ROOT}")
    print(f"Results  : {campaign.results_root}")
    print(f"State    : {STATE_DIR}")
    print(f"Shards   : {len(shards)}")

    if args.dry_run:
        preflight(campaign, executing=False)
        for i, shard in enumerate(shards, 1):
            budget = shard.declared_hours
            budget_txt = "unknown" if budget != budget else f"~{budget:.2f} h"
            print(f"\n[{i}/{len(shards)}] {shard.label}  (declared {budget_txt})")
            print(f"  $ {' '.join(shard_command(campaign, shard))}")
        total = sum(s.declared_hours for s in shards if s.declared_hours == s.declared_hours)
        print(f"\nDeclared total: ~{total:.1f} GPU-h (4-mode baseline; "
              f"this campaign runs {campaign.modes_per_shard} mode(s) per shard)")
        return 0

    preflight(campaign, executing=True)
    args.git_commit = git_commit()
    args.gpu_name = gpu_name()
    print(f"Commit   : {args.git_commit}")
    print(f"GPU      : {args.gpu_name}")
    print(f"Host     : {socket.gethostname()}")
    print(f"Threads  : OMP={os.environ.get('OMP_NUM_THREADS')}")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        return run_campaign(campaign, shards, args)
    except SystemExit as exc:
        # calibration_gate raises SystemExit with a message; surface it cleanly.
        if exc.code not in (0, None):
            print(f"\n{exc.code}", file=sys.stderr)
            return 2
        return 0


if __name__ == "__main__":
    sys.exit(main())
