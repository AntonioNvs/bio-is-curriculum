# Architecture

## Layer diagram

```
YAML / CLI  →  ExperimentConfig  →  pipeline/runner.py
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
     data/loader                  selection/BIOIS              models/modernbert
     preprocessing                (TF-IDF weak clf)            models/logistic_regression
          │                             │                             │
          └─────────────────────────────┴─────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              raw / is           curriculum/methods      baselines/bN
              (single stage)     (phased CL)             (phased or dynamic)
                    │                   │                   │
                    └───────────────────┴───────────────────┘
                                        ▼
                              results/recorder.py
```

## Module boundaries

| Package | Responsibility |
|---------|----------------|
| `config/` | Single source of truth for defaults and YAML loading |
| `data/` | Dataset I/O, val split, rare-class upsampling |
| `selection/` | BIOIS bi-objective instance selection |
| `signals/` | Difficulty scorers shared by CL and baselines |
| `curriculum/` | Internal CL methods (biois_discrete, spcl_*) |
| `baselines/` | Literature CL baselines (b1, b2 SPDCL, …) |
| `models/` | RoBERTa and LR training backends |
| `training/` | Phased vs dynamic training orchestration |
| `pipeline/` | Mode dispatch and single-fold runner |
| `results/` | Artifact recording and cross-fold aggregation |

## Data flow

1. **RoBERTa path**: raw texts from `texts.txt` + aligned TF-IDF recomputed in memory.
2. **BIOIS**: runs on TF-IDF sparse matrix.
3. **RoBERTa training**: runs on raw text strings.
4. **LR smoke tests**: pre-built svmlight TF-IDF files.

## Trainer kinds

| Kind | Used by | API |
|------|---------|-----|
| `phased` | biois_discrete, spcl_*, b1 | `model.fit_stage()` per phase |
| `dynamic` | b2 SPDCL | `model.fit_epoch()` + per-epoch reorder |

## Extension points

- New **curriculum method**: subclass `BIOISCurriculumBase`, register in `curriculum/methods/registry.py`.
- New **baseline**: see [IMPLEMENTING_BASELINES.md](IMPLEMENTING_BASELINES.md).
- New **difficulty signal**: implement in `signals/`.
