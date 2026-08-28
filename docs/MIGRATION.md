# Migration Guide

## Command changes

| Old | New |
|-----|-----|
| `python main.py webkb --mode is_cl` | `uv run bio-run webkb --mode is_cl` |
| `python run.py experiments/webkb.yaml` | `uv run bio-experiment experiments/webkb.yaml` |
| `python run_experiment.py webkb ...` | `uv run bio-experiment experiments/...yaml` |

## Shell scripts → campaign YAML

Removed `scripts/run_docker_*.sh`. Use `bio-experiment` with files in `experiments/campaigns/`:

| Old | New |
|-----|-----|
| `./scripts/run_docker_full_cv.sh 7 webkb` | `uv run bio-experiment experiments/campaigns/full_cv.yaml` |
| `./scripts/run_docker_full_cv_multi.sh 7` | `uv run bio-experiment experiments/campaigns/full_cv_multi.yaml` |
| `./scripts/run_docker_curriculum_cv.sh 7` | `uv run bio-experiment experiments/campaigns/curriculum_cv.yaml` |
| `./scripts/run_docker_large_datasets_5cv.sh 7` | `uv run bio-experiment experiments/campaigns/large_datasets_5cv.yaml` |
| `./scripts/run_docker_smoke_test.sh 7 webkb` | `uv run bio-experiment experiments/campaigns/smoke_docker.yaml --folds 0` |

Legacy shims `main.py` and `run.py` still work.

## Import changes

| Old | New |
|-----|-----|
| `from iSel.biois import BIOIS` | `from bio_is_curriculum.selection.biois import BIOIS` |
| `from curriculum.core import ...` | `from bio_is_curriculum.curriculum.orchestrator import ...` |
| `from results.run import RunRecorder` | `from bio_is_curriculum.results.recorder import RunRecorder` |

## Directory changes

| Old | New |
|-----|-----|
| `src/cli.py` | `src/bio_is_curriculum/pipeline/runner.py` + `cli/` |
| `src/iSel/` | `src/bio_is_curriculum/selection/` |
| `src/curriculum/roberta_model.py` | `src/bio_is_curriculum/models/roberta.py` |
| `run_experiment.py` aggregation | `src/bio_is_curriculum/results/aggregator.py` |

## Mode rename

- `is_continuos_cl` → `is_continuous_cl` (old spelling still accepted as alias).

## Install

```sh
uv sync
```

Package is now installable; no `sys.path` hacks required.

## Results

`results/` is gitignored. Existing local results are preserved on disk.
