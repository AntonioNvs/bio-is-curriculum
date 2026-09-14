# Experiments

Abstract experiment list for the paper. Each block isolates one axis; all share the same setup (datasets, CV, model, metrics).

**Common setup:** text classification with a Transformer (RoBERTa-base), cross-validation, macro-F1 + training time + fraction of data used.

**Execution environment:** define runs in YAML under `experiments/campaigns/` and launch with `uv run bio-experiment …` (Docker + GPU **7** configured in the YAML `docker:` block). See [CONFIGURATION.md](CONFIGURATION.md).

**Core contribution:** curriculum learning guided by BIOIS metrics (redundancy, noise, entropy) — not data selection alone.

---

## 1. Baseline — no IS, no CL

Standard training on the full dataset.

- **Goal:** accuracy and compute reference.
- **Mode:** `raw`

---

## 2. Instance selection only (BIOIS)

Dataset reduction by redundancy and noise, without curriculum.

- **Goal:** measure the isolated effect of instance selection.
- **Mode:** `is`
- **Optional ablation:** redundancy-only vs. noise-only vs. both.

---

## 3. Curriculum learning only (BIOIS signals)

Training organized in phases (easy → hard) using BIOIS metrics as the difficulty signal, without reducing the dataset.

- **Goal:** measure the isolated effect of curriculum with the proposed signal.
- **Mode:** `cl`
- **Internal variants:** BIOIS-discrete (clean → diverse → hard), SPCL soft, SPCL loss

### `biois_discrete` (margin + compute-aware)

`biois_discrete` uses the same 3-phase discrete schedule as the heuristic ablations (`clean → diverse → hard`, per-class quantiles `q_low` / `q_mid` / `q_high`). Signals come from a fitted weak TF-IDF logistic-regression classifier (BIOIS `fitting_alpha`), shared via `signals/biois.py`:

| Signal | Role in curriculum |
|--------|-------------------|
| **Margin `m`** | Label-aware multiclass margin (`P(y|x) - max P(other|x)`), per-class rank normalized (higher = harder). |
| **Entropy `e`** | Bounded Shannon entropy (`H / log K`), per-class rank normalized (higher = harder). |
| **Schedule `s`** | `margin_weight * m + entropy_weight * e`, blended with a `length_weight` word-count prior when texts are available. |
| **Noise `n`** | Deterministic noise risk for *misclassified* samples: `n = 1 - bounded_entropy` (confident mistakes score highest). |
| **Redundancy `r`** | True-class confidence on correct predictions; downweights redundant samples in the hard-phase mid→high slice (`1 - curriculum_beta * min(r, r_cap)`). |

Scheduler (defaults in `curriculum_ablations_multi.yaml`):

1. **Phase ordering:** sort by `max(s, n)` — noise defers confident weak-classifier mistakes out of early phases.
2. **Downweight:** noise penalty `1 - curriculum_beta * n` in phases listed by `noise_weight_phases` (default: **hard only**).
3. **Progressive max length:** train with `96 / 160 / 256` tokens in clean/diverse/hard via `model.set_phase()`; eval still uses full `max_length` (6 epochs total unchanged).

In `cl` mode, BIOIS still runs with `theta = 0` (no stochastic instance removal); noise affects **ordering and weighting only**, not dataset size. Stochastic noise removal for IS remains controlled by `instance_selection.theta` in `is` / `is_cl` modes.

**Rebuild the Docker image** after scheduler changes before launching a campaign:

```sh
docker build -t bio-is-curriculum:latest .
```

### Preliminary results (`curriculum_ablations_multi`)

Fixed discrete schedule (`q = 0.3/0.6/0.95`, `beta = 0.5`, 6 epochs, ModernBERT). Macro-F1 is mean ± half-width of the 95% CI; runtime is mean end-to-end seconds per fold (`total_run_time_s`).

| Dataset | Signal | Macro-F1 | Runtime (s) | Source |
|---------|--------|----------|-------------|--------|
| WebKB | BIO-IS (margin scheduler) | 0.764 ± 0.016 | 297 | `webkb-10cv-20260914-172338_biois_discrete` |
| WebKB | LRC | 0.762 ± 0.015 | 297 | `webkb-10cv-20260911-171345_lrc_discrete` |
| WebKB | Loss / TD | — | — | `20260904-210854` campaign (unchanged) |
| Reuters-90 | BIO-IS (margin scheduler) | 0.386 ± 0.024 | 500 | `reuters90-5cv-20260914-172338_biois_discrete` |
| Reuters-90 | LRC | 0.383 ± 0.013 | 498 | `reuters90-5cv-20260911-171345_lrc_discrete` |
| Reuters-90 | Loss / TD | — | — | `20260904-210854` campaign (unchanged) |
| Yelp-2013 / AG News | all signals | — | — | campaign in progress |

On WebKB and Reuters-90, the margin scheduler closes the gap vs. LRC from the prior noise-only BIO-IS run (`20260911-171345`) and matches or slightly exceeds LRC macro-F1 with lower training time on Reuters. Yelp and AG News BIO-IS runs were still pending at `20260914-172338`.

### Curriculum signal ablations (negative controls)

Same discrete schedule as `biois_discrete` (same `curriculum_q`, phase names, epoch budget). Only the **difficulty signal** changes. These are ablations of the curriculum component, not literature baselines.

Soviany et al. (ACL Insights 2022) show that many heuristic curricula **do not beat random sampling** on BERT/T5 — we compare BIOIS against stronger model-based and linguistically composite alternatives.

| `curriculum.method` | Difficulty signal | Reference | Status |
|---------------------|-------------------|-----------|--------|
| `biois_discrete` | margin + entropy + length prior; noise defer; hard-phase noise/redundancy downweight | proposed | implemented |
| `loss_discrete` | per-sample CE from untrained/pretrained RoBERTa | SPL standard | implemented |
| `lrc_discrete` | LRC composite (length + rarity + sentence-aware Flesch–Kincaid grade) | Ranaldi et al., RANLP 2023 | implemented |
| `td_discrete` | inverse training-dynamics confidence (probe PLM) | Christopoulou et al., EMNLP 2022 | implemented |
| `length_discrete` | sequence length (complexity proxy) | Platanios et al., 2019 | deprecated |
| `tfidf_discrete` | TF-IDF row norm (static lexical complexity) | Soviany et al., 2022 | deprecated |

**Adaptation notes:**

- `td_discrete` uses a short probe fine-tuning run only to score difficulty; the student model still follows the same 3-phase discrete schedule as `biois_discrete` (not the transfer-teacher two-stage setup from the TD-CL paper).
- `lrc_discrete` applies Ranaldi et al.'s LRC composite to **classification documents** (not pre-training sentences). The comprehensibility term uses standard Flesch–Kincaid grade level with per-document sentence counting; length and rarity components are unchanged.

**Key comparison:** `cl` + `biois_discrete` vs. `cl` + `loss_discrete` / `lrc_discrete` / `td_discrete` — does the full BIOIS signal (entropy + noise + redundancy) beat stronger curriculum signals when the scheduling machinery is held fixed?

Run matrix: [`experiments/campaigns/curriculum_ablations_multi.yaml`](../experiments/campaigns/curriculum_ablations_multi.yaml) (`curriculum.method` matrix over 4 datasets).

Launch:

```sh
uv run bio-experiment experiments/campaigns/curriculum_ablations_multi.yaml
```

Background (long runs):

```sh
mkdir -p logs
nohup uv run bio-experiment experiments/campaigns/curriculum_ablations_multi.yaml \
  > logs/curriculum_ablations_multi.log 2>&1 &
```

### CL parameter ablation (`biois_discrete` hyperparameters)

Holds the curriculum **signal** fixed (`biois_discrete`) and varies one scheduler axis per job. Shared defaults match [`curriculum_ablations_multi.yaml`](../experiments/campaigns/curriculum_ablations_multi.yaml); the **reference row** for comparisons is the `biois_discrete` run from that campaign (not re-run here).

| Axis | Job suffix | Focal change | All other params |
|------|------------|--------------|------------------|
| **Schedule** | `_sched_q02-05` | `q_low=0.2`, `q_mid=0.5` | margin 0.6/0.4, progressive lengths |
| **Signal** | `_sig_entropy` | `margin_weight=0.3`, `entropy_weight=0.7` | default quantiles, progressive lengths |
| **Compute** | `_compute_flat` | `phase_max_lengths: [256, 256, 256]` | default quantiles and signal weights |

**Mapping from the pre-margin ablation** (`20260904-232743`):

| Old job | New counterpart |
|---------|-----------------|
| `q03-06_weighted` | reference → `curriculum_ablations_multi` `biois_discrete` defaults |
| `q02-05_weighted` | `_sched_q02-05` |
| `q03-06_unweighted` (`beta=0`) | deferred (optional 4th job) |

Run matrix: [`experiments/campaigns/cl_params_ablation_multi.yaml`](../experiments/campaigns/cl_params_ablation_multi.yaml) — 3 jobs × 4 datasets, `cl` mode only.

```sh
docker build -t bio-is-curriculum:latest .

# Smoke (fold 0, single dataset)
uv run bio-experiment experiments/campaigns/cl_params_ablation_multi.yaml --dataset webkb --folds 0

# Full campaign
uv run bio-experiment experiments/campaigns/cl_params_ablation_multi.yaml
```

Background:

```sh
mkdir -p logs
nohup uv run bio-experiment experiments/campaigns/cl_params_ablation_multi.yaml \
  > logs/cl_params_ablation_multi.log 2>&1 &
```

Compare each job vs. the `curriculum_ablations_multi` reference on macro-F1, hard-slice macro-F1, and `total_run_time_s` via the campaign manifest and `summary.py`.

---

## 4. Instance selection + curriculum learning (proposed method)

BIOIS reduces the dataset; curriculum operates on the subset.

- **Goal:** main result — efficiency with competitive F1.
- **Mode:** `is_cl`
- **CL variants:** discrete, SPCL soft, SPCL loss (same IS, different schedulers)

---

## 5. Curriculum learning baselines (literature)

Comparison with CL methods that **pace or weight instances** using alternative difficulty signals — without BIOIS bi-objective metrics.

- **Goal:** show that CL guided by redundancy + noise + entropy (BIOIS) beats recent CL based on training dynamics or univariate confidence.
- **Scope:** same training budget and phase scheduler when applicable; compare `cl`/`is_cl` (BIOIS) vs. each baseline.

### Foundational (historical reference)

| Baseline | Difficulty signal | Status in repo |
|----------|-------------------|----------------|
| Margin-paced CL (Bengio et al., 2009) | OOF LR multiclass margin (§4.2 proxy) | `b1` |
| Canonical SPCL (Jiang et al., 2015) | region Ψ + reliability prior | `spcl_loss` |

### NLP / fine-tuning — paper priority

Methods designed for PLMs on NLU tasks (classification, NLI, etc.):

| Baseline | Difficulty signal | Reference | Status |
|----------|-------------------|-----------|--------|
| Cross-Review + Annealing (AnnealCR) | teacher-model votes on train subsets | Xu et al., ACL 2020 | to implement |
| Training Dynamics CL (AnnealTD) | uncertainty stats during training (easy / ambiguous / hard) | Christopoulou et al., EMNLP 2022 | `td_discrete` (signal ablation) |
| Competence-based CL | growing model competence (epoch function) | Platanios et al., 2019 | to implement |
| CL-LRC | length + rarity + comprehensibility (LRC) | Ranaldi et al., RANLP 2023 | `lrc_discrete` (signal ablation) |
| Self-adaptive CL | difficulty predicted by the PLM itself | ACL SRW 2025 | to implement |
| SPDCL | linguistic difficulty + dynamic nuclear norm | arXiv 2210.14724 | `b2` implemented |

### Optional (appendix or extension)

| Baseline | Difficulty signal | Note |
|----------|-------------------|------|
| Influence-driven CL | influence of each example on others' loss | pre-training focus; arXiv 2025 |
| Continuous pacing (SPCL soft) | soft pacing over BIOIS signals | already in repo as internal variant |

**Key comparisons for the paper:**

- `is_cl` vs. **AnnealCR** and **AnnealTD** — BIOIS vs. most cited NLU fine-tuning CL methods
- `is_cl` vs. **self-adaptive PLM** — external bi-objective signal vs. Transformer self-reported difficulty
- `cl` + `biois_discrete` vs. `cl` + heuristic ablations — BIOIS beats signals literature considers weak (§3)
- `is_cl` vs. `b1` — gain beyond Bengio-style margin-paced CL (2-phase; differs from `is_cl` 3-phase BIOIS schedule)
- `is_cl` vs. `b2` (SPDCL) — BIOIS vs. dynamic nuclear norm (same epoch budget; see `experiments/spdcl_paper_near.yaml`)
- `raw` vs. `b2` — SPDCL gain over full-data training without IS

---

## 6. Analysis (post-experiments)

Not new training runs; derived from results above.

- Efficiency frontier: macro-F1 vs. training time
- Impact on rare classes
- When the weak-classifier signal transfers to the Transformer
- Case studies: removed vs. kept examples

---

## Summary matrix

| Experiment | IS | CL | Difficulty signal | Role in paper |
|------------|----|----|-------------------|---------------|
| Baseline | ✗ | ✗ | — | Reference |
| Only IS | ✓ | ✗ | — | IS ablation |
| Only CL (BIOIS) | ✗ | ✓ | BIOIS | CL ablation |
| CL signal ablations | ✗ | ✓ | length / loss / TF-IDF | Negative controls (§3) |
| IS + CL (proposed) | ✓ | ✓ | BIOIS | **Main result** |
| SPDCL (`b2`) | ✗ | ✓ | Nuclear norm | NLP literature baseline |
| CL SOTA baselines | ✗/✓ | ✓ | TD, AnnealCR, LRC, PLM… | NLP literature comparison |
| Analysis | — | — | — | Figures and discussion |

---

## Execution priority

1. Baseline + Only IS + Only CL + IS+CL (2² factorial)
2. IS+CL with CL variants (discrete, SPCL soft, SPCL loss)
3. Curriculum signal ablations: `biois_discrete` vs. `loss_discrete` / `lrc_discrete` / `td_discrete`
4. CL parameter ablation: schedule / signal / compute axes (`cl_params_ablation_multi.yaml`)
5. NLP baselines: AnnealCR (ACL 2020) → AnnealTD (EMNLP 2022) → self-adaptive PLM
6. Analyses
