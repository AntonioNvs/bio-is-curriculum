# Baseline Catalog

Stable indices for `--baseline N` and YAML modes `bN`.

| Index | Slug | Name | Signal | Trainer | Paper | Status |
|-------|------|------|--------|---------|-------|--------|
| 1 | `b1` | Margin-paced CL | OOF LR multiclass margin | phased | Bengio et al., ICML 2009 | Implemented |
| 2 | `b2` | SPDCL | Nuclear norm (linguistic + delta) | dynamic | Zhang et al., [arXiv:2210.14724](https://arxiv.org/abs/2210.14724) | Implemented |
| — | `self_adaptive_pmd` | Self-adaptive CL (PMD) | Frozen PLM MLM top-2 margin | PMD batch sampling | Feng et al., ACL SRW 2025 | Implemented (`curriculum.method`) |

Literature baselines registered as `--baseline N` use modes `bN` / `is_bN`. Self-adaptive PMD is invoked as `cl` + `curriculum.method: self_adaptive_pmd` (same fair-comparison harness; no BIOIS).

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

## Self-adaptive CL — PMD (Feng et al., ACL SRW 2025)

Paper: [Your Pretrained Model Tells the Difficulty Itself](https://aclanthology.org/2025.acl-srw.15/) (ACL SRW 2025)

**Paper-style description.** We adapt the self-adaptive curriculum of Feng, Liu & Schütze (ACL SRW 2025). Before any classifier update, a frozen ModernBERT masked language model scores every training example via a cloze prompt and one-token verbalizers; because our labels are numeric, verbalizers are chosen automatically per fold by class-contrastive TF–IDF rather than hand-crafted keywords. Difficulty follows the paper's confidence margin $|P_{\max}-P_{\mathrm{second}}|$ after normalizing over verbalizer tokens (low margin = hard). Fine-tuning then uses the paper's strongest strategy, **PMD**: each batch is partitioned $60{:}40$ into hard- and easy-prioritized draws, with squared-rank multinomial probabilities over the confidence order. We keep the same ModernBERT training budget as our other baselines (6 epochs, batch size 32) so that the comparison isolates the difficulty signal and sampling scheme, not the optimization schedule. Unlike Bengio and SPDCL, this baseline relies on the PLM's own pretrained confidence rather than a weak classifier or training-dynamics geometry, and we evaluate it in full-data `cl` mode (no BIOIS reduction).

### Paper mapping

| Paper element | Implementation |
|---------------|----------------|
| §3.1 cloze prompt + `[MASK]` | `signals/self_adaptive.py` — suffix `" This text is [MASK]."` (configurable) |
| §3.2 one-token verbalizer | Automatic unique tokens via class-contrastive TF–IDF (`select_verbalizers`) |
| §3.3 confidence $\|P_{\max}-P_{\mathrm{second}}\|$ | Frozen `AutoModelForMaskedLM`; normalize over verbalizer logits |
| §3.4.3 PMD ($\|B_1\|{:}\|B_2\|=6{:}4$) | `PMDBatchSampler` in `models/modernbert.py` (`hard_fraction=0.6`) |
| $P(x_n)\propto n^2$ | `pmd_rank_weights` / `rank_exponent=2` |
| Prompt-based fine-tuning | **Not used** — scoring is MLM; student is standard ModernBERT classification |
| Six sampling strategies | **PMD only** (paper's most consistent winner) |

### Algorithm

1. Fit automatic verbalizers on the training split (one eligible vocab token per class).
2. Score all train texts with a **frozen** MLM (no gradient); confidence = top-1 − top-2 class probability.
3. Sort by descending confidence (easy → hard ranks).
4. Train one full-data stage for `epochs` epochs; each batch draws 60% from hard-prioritized squared-rank probs and 40% from the reverse (easy) distribution.

### Hyperparameter mapping (campaign profile)

| Paper (App. A.5) | Our config (`experiments/campaigns/self_adaptive_pmd.yaml`) |
|------------------|--------------------------------------------------------------|
| BERT/RoBERTa-base | `answerdotai/ModernBERT-base` (same checkpoint for MLM score + classifier) |
| 5 epochs, batch 16, lr $1\mathrm{e}{-5}$, no warmup | **6 epochs, batch 32, lr $2\mathrm{e}{-5}$, warmup 0.06** (match SPDCL / curriculum ablations) |
| PMD 6:4 | `curriculum.hard_fraction: 0.6` |
| $P\propto n^2$ | `curriculum.rank_exponent: 2.0` |

### Adaptations (intentional)

- **Automatic verbalizers** — repository labels are numeric (incl. 90 Reuters classes); semantic hand labels are unavailable.
- **ModernBERT MLM scorer** — same `hf_model` as the student; still no-update scoring as in the paper.
- **Classification fine-tuning** — not PET/prompt fine-tuning; only the difficulty signal uses the cloze head.
- **Campaign-matched budget** — fair vs. `b1` / `b2` / `is_cl`, not a paper-hyperparameter reproduction.
- **BIOIS skipped** (`REQUIRES_BIOIS = False`); full-data `cl` only in the published campaign.
- **PMD only** — E2D/D2E/SME/SMD/PME left for optional ablations.

### Config

```yaml
curriculum:
  method: self_adaptive_pmd
  prompt_suffix: " This text is [MASK]."
  hard_fraction: 0.6
  rank_exponent: 2.0
  score_batch_size: 64
training:
  epochs: 6
  epochs_per_phase: 6   # single PMD stage
```

### Requirements

- ModernBERT backend (`model: modernbert`) with raw texts.
- Logs `sa_score_time_s` in `timings.csv`.
- Artifacts: `self_adaptive_verbalizers.json`, `self_adaptive_scores.csv`.

### Results (`self_adaptive_pmd_20260918-192111`)

Macro-F1 and total time are mean ± half-width of the 95% CI (per-fold `total_run_time_s`).

| Dataset | Macro-F1 | Total time (s) |
|---------|----------|----------------|
| WebKB (10-fold) | 0.787 ± 0.016 | 573 ± 4 |
| Reuters-90 (5-fold) | 0.377 ± 0.020 | 828 ± 6 |
| AG News (5-fold) | 0.945 ± 0.001 | 3,843 ± 13 |
| Yelp-2013 (5-fold) | 0.642 ± 0.002 | 20,459 ± 49 |

### Run

```sh
docker build -t bio-is-curriculum:latest .

# Smoke
uv run bio-experiment experiments/campaigns/self_adaptive_pmd.yaml --dataset webkb --folds 0

# Full campaign (GPU 5)
mkdir -p logs
nohup uv run bio-experiment experiments/campaigns/self_adaptive_pmd.yaml \
  > logs/self_adaptive_pmd.log 2>&1 &
```

## Planned baselines

See [EXPERIMENTS.md](EXPERIMENTS.md) for AnnealCR, AnnealTD, length/loss controls, etc.
