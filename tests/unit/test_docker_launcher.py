"""Tests for Docker launcher command construction."""

from bio_is_curriculum.cli.docker_launcher import build_docker_command
from bio_is_curriculum.config.schema import DockerConfig


def test_build_docker_command(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "datasets").mkdir()
    (project / "results").mkdir()
    cfg = project / "experiments" / "test.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("dataset: webkb\nmodes: [raw]\n", encoding="utf-8")

    monkeypatch.setenv("HOST_PROJECT_DIR", str(project))
    docker = DockerConfig(gpu_id=7, cpus=8, memory="16g")
    cmd = build_docker_command(str(cfg), docker, ["--folds", "0", "--no-docker"])

    assert cmd[0] == "docker"
    assert "--gpus=device=7" in cmd
    assert "--cpus=8" in cmd
    assert "--memory=16g" in cmd
    assert "BIO_IS_IN_DOCKER=1" in cmd
    assert f"{project}/datasets:/app/datasets" in cmd
    assert f"{project}/src:/app/src" in cmd
    assert f"{project}/experiments:/app/experiments" in cmd
    assert "python" in cmd
    assert "-m" in cmd
    assert "bio_is_curriculum.cli.experiment" in cmd
    assert "experiments/test.yaml" in cmd
    assert "--no-docker" in cmd
