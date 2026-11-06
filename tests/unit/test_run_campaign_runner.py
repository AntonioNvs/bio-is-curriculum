"""Tests for scripts/run_campaign.py decision logic.

The runner is a script, not a package module, so it is loaded by path. Importing
it must not pull in bio_is_curriculum — that would call configure_cuda_device()
in the test process (see the runner's module docstring). The import itself is
therefore part of what is under test.

These cover the parts where a bug costs GPU-hours silently rather than loudly:
the completion check that gates idempotency, and the failure classification that
decides retry vs poison.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_campaign.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_campaign_under_test", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    # Must be registered before exec_module: @dataclass resolves string
    # annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# --------------------------------------------------------------------------- #
# Helpers to build a fake completed shard on disk
# --------------------------------------------------------------------------- #
def _fake_campaign(results_dir: str = "results") -> "runner.Campaign":
    return runner.Campaign(
        name="fake",
        path=Path("experiments/campaigns/fake.yaml"),
        text="",
        doc={},
        results_dir=results_dir,
        shards=[],
        modes_per_shard=4,
    )


def _write_shard_tree(
    root: Path,
    *,
    modes=("raw", "is", "cl", "is_cl"),
    fold: int = 0,
    run_status: str = "ok",
    drop_artifact: str | None = None,
) -> Path:
    """Create results/experiments/<name>_<ts>/manifest.json + run dirs."""
    exp_id = "books-5cv-20260101-120000_fake"
    run_dir = root / "results" / exp_id
    for mode in modes:
        mode_dir = run_dir / f"{mode}_fold{fold}"
        mode_dir.mkdir(parents=True, exist_ok=True)
        for artifact in runner.REQUIRED_ARTIFACTS:
            if artifact == drop_artifact:
                continue
            (mode_dir / artifact).write_text("x", encoding="utf-8")

    exp_dir = root / "results" / runner.EXPERIMENTS_SUBDIR / "fake_20260101-120000"
    exp_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "runs": [
            {
                "experiment_id": exp_id,
                "path": f"results/{exp_id}",
                "modes": list(modes),
                "status": run_status,
            }
        ]
    }
    manifest_path = exp_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(runner, "DONE_DIR", tmp_path / "state" / "done")
    monkeypatch.setattr(runner, "LEDGER", tmp_path / "state" / "ledger.jsonl")
    monkeypatch.setattr(runner, "HEARTBEAT", tmp_path / "state" / "heartbeat.json")
    monkeypatch.setattr(runner, "LOG_DIR", tmp_path / "logs")
    return tmp_path


# --------------------------------------------------------------------------- #
# Completion / idempotency
# --------------------------------------------------------------------------- #
def test_no_marker_means_not_complete(fake_repo):
    campaign = _fake_campaign()
    shard = runner.Shard(dataset="books", fold=0, declared_hours=1.0)
    complete, why = runner.is_complete(campaign, shard)
    assert complete is False
    assert "marker" in why


def test_complete_shard_is_skipped(fake_repo):
    """The idempotency contract: a verified shard needs no re-run."""
    manifest_path = _write_shard_tree(fake_repo)
    campaign = _fake_campaign()
    shard = runner.Shard(dataset="books", fold=0, declared_hours=1.0)

    runner.mark_done(campaign, shard, {"manifest_path": str(manifest_path)})
    complete, why = runner.is_complete(campaign, shard)
    assert complete is True, why


def test_failed_run_in_manifest_is_not_complete(fake_repo):
    """A manifest is written for failed batches too, so presence is not success."""
    manifest_path = _write_shard_tree(fake_repo, run_status="failed")
    campaign = _fake_campaign()
    shard = runner.Shard(dataset="books", fold=0, declared_hours=1.0)

    ok, why = runner.verify_shard(campaign, shard, {"manifest_path": str(manifest_path)})
    assert ok is False
    assert "not ok" in why


def test_missing_predictions_artifact_is_not_complete(fake_repo):
    """predictions_test.csv is the substrate for the paired bootstrap.

    A run that exits 0 without it is useless for the analysis the campaign
    exists to enable, so it must not be recorded as done.
    """
    manifest_path = _write_shard_tree(fake_repo, drop_artifact="predictions_test.csv")
    campaign = _fake_campaign()
    shard = runner.Shard(dataset="books", fold=0, declared_hours=1.0)

    ok, why = runner.verify_shard(campaign, shard, {"manifest_path": str(manifest_path)})
    assert ok is False
    assert "predictions_test.csv" in why


def test_missing_manifest_is_not_complete(fake_repo):
    campaign = _fake_campaign()
    shard = runner.Shard(dataset="books", fold=0, declared_hours=1.0)
    ok, why = runner.verify_shard(
        campaign, shard, {"manifest_path": str(fake_repo / "nope" / "manifest.json")}
    )
    assert ok is False
    assert "manifest missing" in why


def test_done_marker_is_written_atomically(fake_repo):
    """No leftover .tmp files, and the marker parses."""
    campaign = _fake_campaign()
    shard = runner.Shard(dataset="books", fold=0, declared_hours=1.0)
    runner.mark_done(campaign, shard, {"manifest_path": "x"})

    marker = runner.DONE_DIR / f"{shard.key}.json"
    assert marker.is_file()
    assert json.loads(marker.read_text(encoding="utf-8"))["manifest_path"] == "x"
    assert list(runner.DONE_DIR.glob("*.tmp")) == []


# --------------------------------------------------------------------------- #
# Failure classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code", [130, 137, 143])
def test_preemption_codes_are_retryable(code):
    kind, _ = runner.classify(code, "")
    assert kind == "preempted"


def test_signal_death_is_preemption():
    kind, detail = runner.classify(-9, "")
    assert kind == "preempted"
    assert "signal 9" in detail


def test_python_traceback_is_poison():
    """Deterministic: retrying would reproduce the same traceback."""
    tail = "Traceback (most recent call last):\n  ...\nValueError: boom"
    kind, _ = runner.classify(1, tail)
    assert kind == "poison"


def test_cuda_oom_is_its_own_kind():
    """OOM gets exactly one retry, then poison — not the full retry budget."""
    kind, _ = runner.classify(1, "RuntimeError: CUDA out of memory. Tried to allocate")
    assert kind == "oom"


def test_success_is_ok():
    assert runner.classify(0, "")[0] == "ok"


def test_unknown_exit_code_is_poison_not_retry_loop():
    """An unrecognised failure must not spin. Poison is the bounded choice."""
    assert runner.classify(42, "")[0] == "poison"


# --------------------------------------------------------------------------- #
# Executing a shard (with a trivial stand-in command, so no GPU time is spent)
# --------------------------------------------------------------------------- #
def _stub_command(*args):
    return [sys.executable, "-c", "print('shard ran')"]


def test_run_attempt_streams_output_and_writes_a_heartbeat(fake_repo, monkeypatch):
    monkeypatch.setattr(runner, "shard_command", _stub_command)
    campaign = _fake_campaign()
    shard = runner.Shard("books", 0, 1.0)

    returncode, log_path = runner.run_attempt(campaign, shard, 1)

    assert returncode == 0
    assert "shard ran" in log_path.read_text(encoding="utf-8")
    heartbeat = json.loads(runner.HEARTBEAT.read_text(encoding="utf-8"))
    assert heartbeat["shard"] == "books_f0"
    assert heartbeat["attempt"] == 1


def test_run_attempt_log_is_per_attempt(fake_repo, monkeypatch):
    """Attempts get separate logs so a preemption post-mortem is not overwritten."""
    monkeypatch.setattr(runner, "shard_command", _stub_command)
    campaign = _fake_campaign()
    shard = runner.Shard("books", 0, 1.0)

    _, first = runner.run_attempt(campaign, shard, 1)
    _, second = runner.run_attempt(campaign, shard, 2)
    assert first != second
    assert "attempt1" in first.name and "attempt2" in second.name


def test_failing_attempt_is_classified_from_its_log(fake_repo, monkeypatch):
    monkeypatch.setattr(
        runner, "shard_command",
        lambda *a: [sys.executable, "-c",
                    "import sys; print('Traceback (most recent call last):'); sys.exit(1)"],
    )
    campaign = _fake_campaign()
    shard = runner.Shard("books", 0, 1.0)

    returncode, log_path = runner.run_attempt(campaign, shard, 1)
    kind, _ = runner.classify(returncode, runner.tail_of(log_path))
    assert returncode == 1
    assert kind == "poison"


def test_failed_attempt_output_is_orphaned(fake_repo):
    """Partial dirs carry fresh timestamps and would pollute bio-summary."""
    results = fake_repo / "results"
    pre_existing = results / "books-5cv-old_factorial_matched"
    pre_existing.mkdir(parents=True)
    (pre_existing / "config.json").write_text("{}", encoding="utf-8")

    before = runner.snapshot(results)
    partial = results / "books-5cv-new_factorial_matched"
    partial.mkdir()

    moved = runner.orphan_new_dirs(results, before)

    assert not partial.exists()
    assert (results / runner.ORPHAN_DIRNAME / partial.name).is_dir()
    assert len(moved) == 1
    # Pre-existing results are never touched.
    assert pre_existing.is_dir()


def test_orphaning_twice_does_not_clobber(fake_repo):
    results = fake_repo / "results"
    results.mkdir(parents=True)
    before = runner.snapshot(results)

    for _ in range(2):
        (results / "same_name").mkdir()
        runner.orphan_new_dirs(results, before)

    orphans = sorted(p.name for p in (results / runner.ORPHAN_DIRNAME).iterdir())
    assert orphans == ["same_name", "same_name.1"]


# --------------------------------------------------------------------------- #
# Shard ordering and filtering
# --------------------------------------------------------------------------- #
def test_cheap_first_puts_the_calibration_shard_first():
    shards = [
        runner.Shard("yelp_2013", 0, 13.02),
        runner.Shard("reuters90", 0, 0.54),
        runner.Shard("books", 0, 1.38),
    ]
    ordered = runner.order_shards(shards, "cheap-first")
    assert [s.dataset for s in ordered] == ["reuters90", "books", "yelp_2013"]


def test_big_first_inverts():
    shards = [
        runner.Shard("reuters90", 0, 0.54),
        runner.Shard("yelp_2013", 0, 13.02),
    ]
    ordered = runner.order_shards(shards, "big-first")
    assert [s.dataset for s in ordered] == ["yelp_2013", "reuters90"]


def test_unknown_cost_sorts_last():
    """An unknown cost is not evidence of being cheap."""
    nan = float("nan")
    shards = [
        runner.Shard("mystery", 0, nan),
        runner.Shard("reuters90", 0, 0.54),
    ]
    ordered = runner.order_shards(shards, "cheap-first")
    assert [s.dataset for s in ordered] == ["reuters90", "mystery"]


def test_filter_by_dataset_and_fold():
    shards = [
        runner.Shard("books", f, 1.0) for f in range(3)
    ] + [runner.Shard("agnews", 0, 1.0)]

    assert len(runner.filter_shards(shards, ["books"])) == 3
    assert len(runner.filter_shards(shards, ["books:1"])) == 1
    assert len(runner.filter_shards(shards, ["books:1", "agnews:0"])) == 2


def test_filter_rejects_unknown_dataset():
    shards = [runner.Shard("books", 0, 1.0)]
    with pytest.raises(SystemExit):
        runner.filter_shards(shards, ["nope"])


# --------------------------------------------------------------------------- #
# Campaign loading against the shipped YAMLs
# --------------------------------------------------------------------------- #
def test_y1_loads_25_shards_cheap_first():
    campaign = runner.load_campaign("y1")
    ordered = runner.order_shards(campaign.shards, "cheap-first")

    assert len(ordered) == 25
    assert ordered[0].dataset == "reuters90"
    assert ordered[0].fold == 0
    # The five datasets the runbook scopes Y1 to, medline deferred.
    assert {s.dataset for s in ordered} == {
        "reuters90", "books", "dblp", "agnews", "yelp_2013"
    }
    assert ordered[-1].dataset == "yelp_2013"


def test_y4_is_flagged_as_a_placeholder():
    campaign = runner.load_campaign("y4")
    assert runner.PLACEHOLDER_MARKER in campaign.text


def test_non_placeholder_campaigns_are_clean():
    for name in ("y1", "y2", "y3", "y5"):
        assert runner.PLACEHOLDER_MARKER not in runner.load_campaign(name).text


def test_shard_command_disables_docker_and_uses_the_campaign_path():
    campaign = runner.load_campaign("y1")
    cmd = runner.shard_command(campaign, runner.Shard("books", 3, 1.38))

    assert "--no-docker" in cmd
    assert cmd[-1] == "3"
    assert "--folds" in cmd
    # Relative path: the CLI is run with cwd=REPO_ROOT, and results_dir/data_dir
    # are relative, so an absolute path would still work but a bare filename
    # would not resolve.
    assert "experiments/campaigns/factorial_matched_2x2.yaml" in cmd


def test_modes_per_shard_scales_the_budget():
    """Y1 runs 4 modes per shard; Y2 runs 1, so its declared cost must scale."""
    assert runner.load_campaign("y1").modes_per_shard == 4
    assert runner.load_campaign("y2").modes_per_shard == 1
