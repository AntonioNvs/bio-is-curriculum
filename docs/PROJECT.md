# Project Overview

## Goal

Fine-tune **ModernBERT-base** on text classification benchmarks using **Bi-Objective Instance Selection (BIOIS)** and **Curriculum Learning (CL)** to improve the **efficiency–quality Pareto frontier**.

## Core hypothesis

Weak-classifier signals (redundancy + noise/entropy from BIOIS) can:

1. **Prune** redundant/noisy training instances (~30–40% reduction on large datasets).
2. **Pace** difficulty during PLM fine-tuning (easy → hard phases).

Together, these yield competitive macro-F1 with less data and less training time vs. standard fine-tuning.

## Experiment matrix (2² factorial)

| Mode | Instance selection | Curriculum | Role |
|------|-------------------|------------|------|
| `raw` | No | No | Reference |
| `is` | Yes | No | IS ablation |
| `cl` | Signals only | Yes | CL ablation |
| `is_cl` | Yes | Yes | **Main method** |

Literature baselines (`b1`, `b2`, …) use alternative difficulty signals/schedules under the same training budget.

## Relation to prior work

- **SIGIR'23 BIOIS paper (Cunha et al.)**: instance selection for Transformer efficiency.
- **This repo**: reuses BIOIS signals to **schedule curriculum phases** during RoBERTa fine-tuning — a distinct contribution from IS alone.

## Execution

Experiments are YAML-first: see [`experiments/campaigns/`](../experiments/campaigns/) and [CONFIGURATION.md](CONFIGURATION.md).

```sh
docker build -t bio-is-curriculum:latest .
uv run bio-experiment experiments/campaigns/full_cv.yaml
```

## Primary datasets

| Dataset | Size | Claim role |
|---------|------|------------|
| `webkb`, `reuters90` | Small | Fast ablation / debugging |
| `yelp_2013`, `agnews`, `medline` | Large | Efficiency claim (5-fold CV) |

Download: `uv run python download_datasets.py webkb yelp_2013 agnews`

## Key metrics

- **macro-F1** (primary)
- **train_time_s**, **data_efficiency**, **efficiency_score** (from `summary.csv`)
