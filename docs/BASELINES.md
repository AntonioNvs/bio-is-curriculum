# Baseline Catalog

Stable indices for `--baseline N` and YAML modes `bN`.

| Index | Slug | Name | Signal | Trainer | Paper | Status |
|-------|------|------|--------|---------|-------|--------|
| 1 | `b1` | Confidence-paced CL | Weak-LR label confidence | phased | Bengio et al., ICML 2009 | Implemented |
| 2 | `b2` | SPDCL | Nuclear norm (linguistic + delta) | dynamic | Zhang et al., [arXiv:2210.14724](https://arxiv.org/abs/2210.14724) | Implemented |

## b1 — Bengio 2009

- **Signal**: confidence of weak LR on true label (`BIOIS._probaEveryone`).
- **Schedule**: cumulative quantile phases (easy → medium → all).
- **Fair comparison**: same `curriculum_q` quantiles as `is_cl`.

## b2 — SPDCL (Zhang et al. 2022)

Paper: [Improving Imbalanced Text Classification with Dynamic Curriculum Learning](https://arxiv.org/abs/2210.14724)

### Algorithm 1 mapping

| Paper step | Implementation |
|------------|----------------|
| Nuclear norm on all token hidden states | `RobertaModel.extract_hidden_states()` + `NuclearNormScorer` |
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

- RoBERTa backend (`--model roberta`).
- Logs `nuclear_norm_time_s` (accumulated) in `timings.csv`.

### Run

```sh
uv run bio-experiment experiments/spdcl_smoke.yaml      # fast webkb smoke
uv run bio-experiment experiments/spdcl_paper_near.yaml  # yelp_2013 fold 0
uv run bio-run webkb --baseline 2 --fold 0 --experiment-id my-b2
```

## Planned baselines

See [EXPERIMENTS.md](EXPERIMENTS.md) for AnnealCR, AnnealTD, length/loss controls, etc.
