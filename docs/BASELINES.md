# Baseline Catalog

Stable indices for `--baseline N` and YAML modes `bN`.

| Index | Slug | Name | Signal | Trainer | Paper | Status |
|-------|------|------|--------|---------|-------|--------|
| 1 | `b1` | Margin-paced CL | OOF LR multiclass margin | phased | Bengio et al., ICML 2009 | Implemented |
| 2 | `b2` | SPDCL | Nuclear norm (linguistic + delta) | dynamic | Zhang et al., [arXiv:2210.14724](https://arxiv.org/abs/2210.14724) | Implemented |

## b1 — Bengio 2009

Paper: [Curriculum Learning](https://doi.org/10.1145/1553374.1553380) (ICML 2009)

### Paper mapping

| Paper element | Implementation |
|---------------|----------------|
| §3 embedded sets `Q_λ` | Cumulative easy → full (2 phases) |
| §4.2 margin easiness | `P(y) - max P(c≠y)` from OOF TF-IDF LR |
| §5 switch epoch (~50% budget on easy) | 2 phases × equal `epochs_per_phase` |
| §4.2 oracle `w*` | `signals/oracle_margin.py` (standalone; **no BIOIS**) |
| PLM fine-tuning | RoBERTa/LR (adaptation; paper uses shallow nets/SGD) |
| Rare-class pinning | `balance_phase_indices` safeguard for imbalanced text |

### Algorithm

1. Score each example with OOF 5-fold LR on TF-IDF: `margin = P(y) - max_{c≠y} P(c)`.
2. **Phase `easy`**: top `b1_easy_fraction` by global margin (default 0.5).
3. **Phase `target`**: all training examples (warm-start from phase 1).

Uniform sample weights; no instance selection, redundancy, or entropy weighting.

### Hyperparameter mapping (paper-near profile)

| Paper (§5 shape experiment) | Our config (`experiments/bengio_paper_near.yaml`) |
|-----------------------------|-----------------------------------------------------|
| ~50% epochs on easy domain | `epochs_per_phase: 3` × 2 phases = 6 total |
| easy subset then target | `baseline.b1_easy_fraction: 0.5` |
| fixed pacing | 2 discrete phases |

### Adaptations (intentional)

- **Multiclass margin proxy** for §4.2 oracle margin on text (paper uses known `w*` or separate datasets).
- **Global easy fraction** instead of per-class quantiles (BIOIS/`is_cl` use per-class stratification).
- **2 phases** (not 3 like `is_cl`) — closer to §5 two-stage schedule.
- **BIOIS not run** for pure `b1` (timing: `b1_margin_score_time_s`).

### Config

```yaml
baseline:
  b1_easy_fraction: 0.5
  b1_use_global_quantile: true   # false → per-class legacy stratification
```

### Run

```sh
uv run bio-experiment experiments/bengio_paper_near.yaml
uv run bio-run webkb --baseline 1 --fold 0 --experiment-id my-b1
```

## b2 — SPDCL (Zhang et al. 2022)

Paper: [Improving Imbalanced Text Classification with Dynamic Curriculum Learning](https://arxiv.org/abs/2210.14724)

### Algorithm 1 mapping

| Paper step | Implementation |
|------------|----------------|
| Nuclear norm on all token hidden states | `ModernBertModel.extract_hidden_states()` + `NuclearNormScorer` |
| Epoch 1: sort ascending (easy → hard) | `curriculum_epoch == 0`, cached `initial_norms` |
| Epoch t>1: sort by descending delta | `score_delta()`, `argsort(-difficulty)` |
| Interleaved scatter into k bins | `scatter_into_bins()` |
| Progressive bin union | `progressive_bin_indices(bins, epoch)` |
| Full-data anneal | `anneal_epochs` after `curriculum_epochs` |

### Hyperparameter mapping (paper-near profile)

| Paper (BERT-base) | Our config (`experiments/spdcl_paper_near.yaml`) |
|-------------------|--------------------------------------------------|
| batch=25 | `training.batch_size: 25` |
| max_length=250 | `training.max_length: 250` |
| lr=5e-5 | `training.lr: 5.0e-5` |
| k bins | `baseline.spdcl_n_bins: 5` |
| curriculum + anneal | `spdcl_curriculum_epochs: 5`, `spdcl_anneal_epochs: 1` |

### Adaptations (intentional)

- **RoBERTa-base** instead of BERT-base (paper cites RoBERTa as compatible).
- **Zenodo single-label** datasets (`yelp_2013`, `webkb`) instead of AAPD/MRPC/CoLA.
- **Multi-label AAPD** out of scope for v1.
- **90/10 stratified val** (seed 2018) for logging; paper Algorithm 1 has no val split.
- **BIOIS skipped** for b2 (paper does not use weak-classifier signals).
- **`inverse_freq_ce`** imbalance loss (our imbalanced-data adaptation).

### Config

```yaml
baseline:
  spdcl_n_bins: 5
  spdcl_curriculum_epochs: 5   # default: n_bins
  spdcl_anneal_epochs: 1
  spdcl_norm_subsample: null   # dev only: subsample norm computation
```

### Requirements

- ModernBERT backend (`--model modernbert`).
- Logs `nuclear_norm_time_s` (accumulated) in `timings.csv`.

### Run

```sh
uv run bio-experiment experiments/spdcl_smoke.yaml      # fast webkb smoke
uv run bio-experiment experiments/spdcl_paper_near.yaml  # yelp_2013 fold 0
uv run bio-run webkb --baseline 2 --fold 0 --experiment-id my-b2
```

## Planned baselines

See [EXPERIMENTS.md](EXPERIMENTS.md) for AnnealCR, AnnealTD, length/loss controls, etc.
