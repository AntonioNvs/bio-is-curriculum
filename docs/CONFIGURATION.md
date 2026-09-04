# Configuration

All hyperparameters flow through `ExperimentConfig` (`src/bio_is_curriculum/config/schema.py`).

Defaults live in `config/defaults.py` — **do not duplicate** defaults in CLI or YAML merge logic.

## Running experiments (standard)

Experiments are defined in YAML and launched with `bio-experiment`. Files under [`experiments/campaigns/`](../experiments/campaigns/) include a `docker:` block — the runner automatically wraps execution in Docker on the host.

```sh
# Full CV on host GPU 7 (docker block in YAML)
uv run bio-experiment experiments/campaigns/full_cv.yaml

# Smoke test, single fold
uv run bio-experiment experiments/campaigns/smoke_docker.yaml --folds 0

# Override GPU / dry-run
uv run bio-experiment experiments/campaigns/full_cv_multi.yaml --docker-gpu 7 --dry-run

# Inside container (after docker build)
uv run bio-experiment experiments/campaigns/full_cv.yaml --no-docker
```

| Campaign YAML | Replaces (removed) |
|---------------|-------------------|
| `campaigns/full_cv.yaml` | `run_docker_full_cv.sh` |
| `campaigns/full_cv_multi.yaml` | `run_docker_full_cv_multi.sh` |
| `campaigns/curriculum_cv.yaml` | `run_docker_curriculum_cv.sh` |
| `campaigns/large_datasets_5cv.yaml` | `run_docker_large_datasets_5cv.sh` |
| `campaigns/smoke_docker.yaml` | `run_docker_smoke_test.sh` |

## Simple batch YAML

Single dataset, one experiment id:

```yaml
docker:
  image: bio-is-curriculum:latest
  gpu_id: 7
  cpus: 16
  memory: 32g

dataset: webkb
n_splits: 10
model: modernbert
hf_model: answerdotai/ModernBERT-base

modes: [raw, is, cl, is_cl, b1, b2]

instance_selection:
  beta: 0.3
  theta: 0.2

curriculum:
  method: biois_discrete
  q_low: 0.3
  q_mid: 0.6
  q_high: 0.95

training:
  epochs: 6
  epochs_per_phase: 2
  train_fraction: 1.0   # fraction of train split (val/test unchanged); e.g. 0.1 for smoke
  batch_size: 32
  max_length: 256
  lr: 2.0e-5
  imbalance_method: inverse_freq_ce
```

## Campaign YAML (matrix jobs)

Multi-dataset sweeps with shared defaults and Cartesian matrix expansion:

```yaml
docker:
  gpu_id: 7

campaign:
  timestamp: auto
  datasets:
    webkb: { n_splits: 10 }
    reuters90: { n_splits: 5 }

  defaults:
    model: modernbert
    hf_model: answerdotai/ModernBERT-base
    training: { epochs: 6, epochs_per_phase: 2 }

  jobs:
    - modes: [raw, is, b1]
      experiment_id: "{dataset}-{n_splits}cv-{timestamp}"

    - modes: [cl, is_cl]
      matrix:
        curriculum.method: [biois_discrete, spcl_soft]
      experiment_id: "{dataset}-{n_splits}cv-{timestamp}_{method}"

    - modes: [cl, is_cl]
      matrix:
        curriculum.method: [spcl_loss]
        curriculum.loss_scheme: [linear, mixture]
      experiment_id: "{dataset}-{n_splits}cv-{timestamp}_{method}_{loss_scheme}"
```

**Template variables:** `{dataset}`, `{n_splits}`, `{timestamp}`, `{method}`, `{loss_scheme}`, and short names from matrix keys.

Optional campaign metadata:

```yaml
campaign:
  name: curriculum_ablations_multi   # used in manifest filename
  timestamp: auto
  summary:
    layout: compare_by_dataset       # or long_table, or auto
    metrics: [macro_f1, micro_f1, hard_slice_macro_f1, train_time_s, total_time]
    datasets: null                   # null = all datasets in the run
  datasets:
    webkb: { n_splits: 10 }
  jobs: [...]
```

Simple batch YAMLs can define a top-level `summary:` block with the same fields.

## Experiment manifest and summary export

Every `bio-experiment` invocation writes a manifest folder:

```
results/experiments/<event_description>_<timestamp>/
    manifest.json
    summary.xlsx    # after running summary.py
    summary.csv
```

Example after a campaign:

```sh
uv run bio-experiment experiments/campaigns/curriculum_ablations_multi.yaml --folds 0
# → results/experiments/curriculum_ablations_multi_20260828-014706/manifest.json

uv run python summary.py results/experiments/curriculum_ablations_multi_20260828-014706/
# → .../summary.xlsx and .../summary.csv
```

Or use the console entry point: `uv run bio-summary results/experiments/<event>_<timestamp>/`

Manifest `summary.layout`:

| Layout | Excel sheets |
|--------|----------------|
| `compare_by_dataset` | one worksheet per dataset, wide metric columns |
| `long_table` | single worksheet, all runs in long format |

Legacy folder discovery (`--run-prefix`, explicit folder args) still works but is deprecated.

## SPDCL epoch budget

Total training epochs for `b2`:

```
total = spdcl_curriculum_epochs (or n_bins if null) + spdcl_anneal_epochs
```

Paper-near profile: `experiments/spdcl_paper_near.yaml` (5 + 1 = 6 epochs).

## Curriculum methods

| Method | Difficulty signal | Requires BIOIS |
|--------|-------------------|----------------|
| `biois_discrete` | BIOIS entropy (+ redundancy in hard phase) | yes |
| `loss_discrete` | per-sample CE (untrained RoBERTa forward pass) | no |
| `lrc_discrete` | LRC composite (length + rarity + readability) | no |
| `td_discrete` | inverse probe-epoch confidence (`td_probe_epochs`, default 2) | no |
| `length_discrete` | sequence word count (deprecated) | no |
| `tfidf_discrete` | TF-IDF row L2 norm (deprecated) | no |
| `spcl_soft` | BIOIS + soft pacing | yes |
| `spcl_loss` | BIOIS + SPCL Algorithm 1 | yes |

`td_discrete` config (under `curriculum:`):

```yaml
td_probe_epochs: 2
td_metric: confidence   # or variability
```

## GPU device

| Context | How GPU is chosen |
|---------|-------------------|
| **Host + `docker:` in YAML** | `docker.gpu_id` (default **7**) → `docker run --gpus device=N` |
| **Inside container** | Single visible GPU as `cuda:0`; Python does not set `CUDA_VISIBLE_DEVICES` |
| **Bare-metal** | `cuda_device_id: 7` or `--cuda-device-id` |

CLI: `--docker-gpu N` overrides `docker.gpu_id`. `--docker` forces Docker wrap even without a `docker:` block.

## Modes

| Token | Meaning |
|-------|---------|
| `raw`, `is`, `cl`, `is_cl` | Core factorial modes |
| `is_continuous_cl` | Alias for IS+CL with `spcl_soft` |
| `b1`, `b2`, … | Literature baselines |

## CLI flags (`bio-experiment`)

| Flag | Purpose |
|------|---------|
| `--folds 0 1` | Subset of folds |
| `--dataset webkb` | Filter campaign jobs or override simple batch dataset |
| `--fail-fast` | Stop on first failure |
| `--dry-run` | Print expanded jobs / docker command |
| `--no-docker` | Run locally (inside container) |
| `--docker` | Force Docker wrap |

Single-fold debugging: `uv run bio-run webkb --fold 0 --mode is_cl`

## Schema version

`config.json` includes `schema_version: "1.0.0"` for reproducibility tracking.
