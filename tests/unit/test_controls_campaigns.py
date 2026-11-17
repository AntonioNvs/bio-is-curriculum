"""Tests over the shipped control-campaign YAMLs.

These read the real files under experiments/campaigns/ rather than inline YAML,
so the artifacts the runbook depends on cannot drift silently. The loader has no
unknown-key validation (see docs/experimentos_maquina.md), so a typo in a
campaign file falls back to DEFAULTS with no error — exactly the failure mode
these tests exist to catch.
"""

from pathlib import Path

from bio_is_curriculum.config.campaign import expand_campaign
from bio_is_curriculum.config.loader import load_experiment_spec

CAMPAIGNS = Path(__file__).resolve().parents[2] / "experiments" / "campaigns"
TIMESTAMP = "20260101-120000"


def _expand(name: str):
    spec = load_experiment_spec(str(CAMPAIGNS / name))
    assert spec.campaign is not None
    return expand_campaign(spec.campaign, timestamp=TIMESTAMP)


def test_y3_expands_to_four_arms_per_dataset():
    batches = _expand("controls_signal_same_selection.yaml")
    # 3 datasets x 4 jobs (composite, margin-only, entropy-only, random phases)
    assert len(batches) == 12

    arms = {b.experiment_id.rsplit("_sel_", 1)[-1] for b in batches}
    assert arms == {"composite", "margin_only", "entropy_only", "random_phases"}

    for batch in batches:
        assert batch.modes == ["is_cl"]
        assert batch.folds == [0, 1, 2, 3, 4]


def test_y3_freezes_selection_across_arms():
    """All four arms must share beta/theta/random_state, or the control is void.

    The whole point is that BIOIS picks the same survivors in every arm, so the
    difficulty score is the only variable.
    """
    batches = _expand("controls_signal_same_selection.yaml")
    signatures = {
        (b.run.beta, b.run.theta, b.run.random_state, b.run.dataset) for b in batches
    }
    # One signature per dataset, not per arm.
    assert len(signatures) == 3
    assert {s[:3] for s in signatures} == {(0.3, 0.2, 42)}


def test_y3_random_phases_method_reaches_the_config():
    """curriculum.method: [random_discrete] must survive into ExperimentConfig.

    Guards the silent-key failure mode: if the key were misspelled the loader
    would keep the default biois_discrete and the arm would silently duplicate
    the composite reference instead of testing random ordering.
    """
    batches = _expand("controls_signal_same_selection.yaml")
    random_arms = [b for b in batches if b.experiment_id.endswith("_sel_random_phases")]

    assert len(random_arms) == 3
    for batch in random_arms:
        assert batch.run.curriculum_method == "random_discrete"


def test_y3_score_arms_carry_their_weights():
    batches = _expand("controls_signal_same_selection.yaml")
    by_arm = {b.experiment_id.rsplit("_sel_", 1)[-1]: b for b in batches if "books" in b.experiment_id}

    assert by_arm["composite"].run.curriculum_method == "biois_discrete"
    assert (by_arm["composite"].run.curriculum_margin_weight,
            by_arm["composite"].run.curriculum_entropy_weight) == (0.6, 0.4)

    assert (by_arm["margin_only"].run.curriculum_margin_weight,
            by_arm["margin_only"].run.curriculum_entropy_weight) == (1.0, 0.0)

    assert (by_arm["entropy_only"].run.curriculum_margin_weight,
            by_arm["entropy_only"].run.curriculum_entropy_weight) == (0.0, 1.0)

    # The random-phases arm keeps the composite weights: only the ordering score
    # changes, so its margin/entropy inputs must stay identical to the reference.
    assert (by_arm["random_phases"].run.curriculum_margin_weight,
            by_arm["random_phases"].run.curriculum_entropy_weight) == (0.6, 0.4)


def test_y3_quantiles_are_not_dropped():
    """q_low/q_mid/q_high must reach the config for every arm, including random."""
    batches = _expand("controls_signal_same_selection.yaml")
    for batch in batches:
        assert tuple(batch.run.curriculum_q) == (0.3, 0.6, 0.95)


def test_y1_excludes_medline_and_the_raw_epoch_job():
    """Y1 is scoped to five datasets and one job; medline is deferred by design."""
    batches = _expand("factorial_matched_2x2.yaml")
    datasets = {b.dataset for b in batches}
    assert datasets == {"reuters90", "books", "dblp", "agnews", "yelp_2013"}
    assert "medline" not in datasets

    # 5 datasets x 1 job = 5 batches, each with all four modes -> 25 shards of 4.
    assert len(batches) == 5
    for batch in batches:
        assert batch.modes == ["raw", "is", "cl", "is_cl"]


def test_y2_covers_three_raw_budgets():
    batches = _expand("raw_curve_diagnostic.yaml")
    assert len(batches) == 6  # 3 datasets x 2 epoch budgets
    assert {b.run.epochs for b in batches} == {2, 4}
    for batch in batches:
        assert batch.modes == ["raw"]
        assert batch.run.curriculum_method == "biois_discrete"


def test_y5_drops_dblp():
    batches = _expand("beta_theta_sweep.yaml")
    assert {b.dataset for b in batches} == {"reuters90", "books"}
    assert len(batches) == 18  # 2 datasets x 9 grid cells
    for batch in batches:
        assert batch.folds == [0, 1, 2]


def test_y4_expands_to_one_run_per_dataset_at_its_measured_retention():
    """The random-subsample control, filled in from Y1's measured retention.

    One job per dataset rather than one shared matrix value: the retention
    genuinely differs by dataset, and a single constant would give two of the
    three controls the wrong denominator — silently, since the run still
    completes and only the comparison is void.
    """
    batches = _expand("controls_random_stratified.yaml")

    assert len(batches) == 3  # 3 datasets x 1 job x 1 matrix value
    assert {b.dataset for b in batches} == {"books", "reuters90", "agnews"}

    measured = {"reuters90": 0.6370, "books": 0.6596, "agnews": 0.6831}
    for batch in batches:
        assert batch.modes == ["raw"]
        assert batch.folds == [0, 1, 2, 3, 4]
        assert batch.run.train_fraction == measured[batch.dataset]


def test_y4_is_no_longer_a_placeholder():
    """The runner refuses any campaign still carrying the marker.

    Y4 is runnable now, so the marker must be gone — and it must not creep back
    in by being quoted in the file's own prose, which would block the campaign
    just as effectively as leaving it in the job.
    """
    text = (CAMPAIGNS / "controls_random_stratified.yaml").read_text(encoding="utf-8")
    assert "TODO(placeholder)" not in text
