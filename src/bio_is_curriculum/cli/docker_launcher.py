"""Launch bio-experiment inside Docker from the host."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from bio_is_curriculum.config.schema import DockerConfig

# Container Python (Dockerfile: PATH includes /app/.venv/bin). Do not use the
# host venv path from sys.executable — it does not exist inside the container.
_CONTAINER_PYTHON = "python"


def _project_root() -> Path:
    env_root = os.environ.get("HOST_PROJECT_DIR")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


def _container_config_path(config_path: str, project_root: Path) -> str:
    path = Path(config_path).resolve()
    try:
        rel = path.relative_to(project_root)
        return rel.as_posix()
    except ValueError:
        return path.as_posix()


def build_docker_command(
    config_path: str,
    docker: DockerConfig,
    inner_argv: list[str],
) -> list[str]:
    """Build ``docker run`` argv for an inner ``bio-experiment`` invocation."""
    project_root = Path(docker.host_project_dir or _project_root()).resolve()
    container_cfg = _container_config_path(config_path, project_root)
    workdir = docker.container_workdir.rstrip("/")

    datasets_mount = f"{project_root}/datasets:{workdir}/datasets"
    results_mount = f"{project_root}/results:{workdir}/results"
    src_mount = f"{project_root}/src:{workdir}/src"
    experiments_mount = f"{project_root}/experiments:{workdir}/experiments"

    cmd = [
        "docker",
        "run",
        "--rm",
        f"--gpus=device={docker.gpu_id}",
        f"--cpus={docker.cpus}",
        f"--memory={docker.memory}",
        "-e",
        "BIO_IS_IN_DOCKER=1",
        "-e",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "-e",
        "PYTHONHASHSEED=42",
        "-e",
        f"OMP_NUM_THREADS={docker.cpus}",
        "-e",
        f"MKL_NUM_THREADS={docker.cpus}",
        "-v",
        datasets_mount,
        "-v",
        results_mount,
        "-v",
        src_mount,
        "-v",
        experiments_mount,
        "-w",
        workdir,
        docker.image,
        _CONTAINER_PYTHON,
        "-m",
        "bio_is_curriculum.cli.experiment",
        container_cfg,
        "--no-docker",
        *inner_argv,
    ]
    return cmd


def docker_run(
    config_path: str,
    docker: DockerConfig,
    inner_argv: list[str] | None = None,
    *,
    dry_run: bool = False,
) -> int:
    """Run ``bio-experiment`` inside Docker; return subprocess exit code."""
    cmd = build_docker_command(config_path, docker, inner_argv or [])
    if dry_run:
        print(" ".join(cmd))
        return 0
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)
