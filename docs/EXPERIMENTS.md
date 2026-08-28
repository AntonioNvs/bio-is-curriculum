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

### Curriculum signal ablations (negative controls)

Same discrete schedule as `biois_discrete` (same `curriculum_q`, phase names, epoch budget). Only the **difficulty signal** changes. These are ablations of the curriculum component, not literature baselines.

Soviany et al. (ACL Insights 2022) show that many heuristic curricula **do not beat random sampling** on BERT/T5 — they position BIOIS against weak alternatives.

| `curriculum.method` | Difficulty signal | Reference | Status |
|---------------------|-------------------|-----------|--------|
| `biois_discrete` | BIOIS entropy (+ redundancy weighting in hard phase) | proposed | implemented |
| `length_discrete` | sequence length (complexity proxy) | Platanios et al., 2019 | implemented |
| `loss_discrete` | per-sample CE from untrained/pretrained RoBERTa | SPL standard | implemented |
| `tfidf_discrete` | TF-IDF row norm (static lexical complexity) | Soviany et al., 2022 | implemented |

**Key comparison:** `cl` + `biois_discrete` vs. `cl` + `length_discrete` / `loss_discrete` / `tfidf_discrete` — does BIOIS entropy beat common heuristics when the scheduling machinery is held fixed?

Run matrix: `experiments/curriculum_ablations.yaml` (swap `curriculum.method` per run).

Launch: `uv run bio-experiment experiments/curriculum_ablations.yaml` (add `docker:` block or use `--docker`).

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
| Confidence-paced CL (Bengio et al., 2009) | label confidence (weak classifier) | `b1` |
| Canonical SPCL (Jiang et al., 2015) | region Ψ + reliability prior | `spcl_loss` |

### NLP / fine-tuning — paper priority

Methods designed for PLMs on NLU tasks (classification, NLI, etc.):

| Baseline | Difficulty signal | Reference | Status |
|----------|-------------------|-----------|--------|
| Cross-Review + Annealing (AnnealCR) | teacher-model votes on train subsets | Xu et al., ACL 2020 | to implement |
| Training Dynamics CL (AnnealTD) | uncertainty stats during training (easy / ambiguous / hard) | Christopoulou et al., EMNLP 2022 | to implement |
| Competence-based CL | growing model competence (epoch function) | Platanios et al., 2019 | to implement |
| CL-LRC | length + rarity + comprehensibility (LRC) | Ranaldi et al., RANLP 2023 | to implement |
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
- `is_cl` vs. `b1` — gain beyond classic confidence-paced CL
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
3. Curriculum signal ablations: `biois_discrete` vs. `length_discrete` / `loss_discrete` / `tfidf_discrete`
4. NLP baselines: AnnealCR (ACL 2020) → AnnealTD (EMNLP 2022) → self-adaptive PLM
5. Analyses
