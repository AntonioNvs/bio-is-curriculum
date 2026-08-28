"""Tests for campaign YAML loading and matrix expansion."""

from bio_is_curriculum.config.campaign import expand_campaign
from bio_is_curriculum.config.loader import load_experiment_spec


CAMPAIGN_YAML = """
docker:
  gpu_id: 7
campaign:
  timestamp: "20260101-120000"
  datasets:
    webkb: { n_splits: 10 }
    reuters90: { n_splits: 5 }
  defaults:
    n_splits: 10
    model: roberta
    training:
      epochs: 6
  jobs:
    - modes: [raw, is]
      experiment_id: "{dataset}-{n_splits}cv-{timestamp}"
    - modes: [cl]
      matrix:
        curriculum.method: [biois_discrete, spcl_soft]
      experiment_id: "{dataset}-{n_splits}cv-{timestamp}_{method}"
"""


def test_load_simple_batch(tmp_path):
    path = tmp_path / "simple.yaml"
    path.write_text(
        "dataset: webkb\nmodes: [raw]\ntraining:\n  epochs: 2\n",
        encoding="utf-8",
    )
    spec = load_experiment_spec(str(path))
    assert spec.batch is not None
    assert spec.campaign is None
    assert spec.batch.dataset == "webkb"


def test_load_campaign_with_docker(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN_YAML, encoding="utf-8")
    spec = load_experiment_spec(str(path))
    assert spec.campaign is not None
    assert spec.docker is not None
    assert spec.docker.gpu_id == 7


def test_expand_campaign_matrix(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN_YAML, encoding="utf-8")
    spec = load_experiment_spec(str(path))
    batches = expand_campaign(spec.campaign)
    # 2 datasets × (1 non-matrix job + 2 curriculum methods) = 6
    assert len(batches) == 6
    ids = {b.experiment_id for b in batches}
    assert "webkb-10cv-20260101-120000" in ids
    assert "webkb-10cv-20260101-120000_biois_discrete" in ids
    assert "reuters90-5cv-20260101-120000_spcl_soft" in ids

    cl_batches = [b for b in batches if "biois_discrete" in (b.experiment_id or "")]
    assert cl_batches[0].run.curriculum_method == "biois_discrete"
    assert cl_batches[0].modes == ["cl"]
