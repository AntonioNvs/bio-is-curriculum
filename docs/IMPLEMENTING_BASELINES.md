# Implementing Baselines

Step-by-step guide for adding a literature curriculum-learning baseline.

## 1. Choose trainer kind

```
Does the method re-sort/resample EVERY epoch?
├── Yes → dynamic (like SPDCL b2)
└── No  → phased (like Bengio b1)
```

## 2. Define difficulty signal

Implement or reuse a scorer in `signals/`:

```python
# signals/my_signal.py
def score(...)-> np.ndarray:
    ...  # higher = harder
```

Existing signals:
- `signals/biois.py` — weak-classifier redundancy + entropy
- `signals/nuclear_norm.py` — SPDCL nuclear norm
- `signals/heuristics.py` — length (curriculum ablation)
- `signals/lexical.py` — TF-IDF rank (curriculum ablation)
- `signals/loss.py` — per-sample CE loss (curriculum ablation)

**Note:** length, loss, and TF-IDF controls are **curriculum signal ablations** (`curriculum.method: length_discrete`, etc.), not literature baselines. See [EXPERIMENTS.md](EXPERIMENTS.md) §3.

## 3. Implement schedule

**Phased** (extend `BaselineBase`):
- Override `_extract_signals()` and `_build_phases()`.

**Dynamic** (extend `DynamicBaselineBase`):
- Implement `fit()` using `training/dynamic.py` or custom epoch loop.

## 4. Register baseline

```python
# baselines/bN_my_method.py
class BaselineN(DynamicBaselineBase):  # or BaselineBase
    INDEX = N
    NAME = "..."
    REFERENCE = "..."
```

Add to `baselines/__init__.py` `REGISTRY`.

## 5. Add config fields

Extend `ExperimentConfig` in `config/schema.py` and `config/defaults.py` if new hyperparameters are needed.

## 6. Add experiment YAML

```yaml
modes: [bN]
baseline:
  my_param: 0.5
```

## 7. Validate

```sh
uv run pytest tests/unit/test_my_baseline.py -q
uv run bio-run webkb --fold 0 --baseline N --model lr --epochs 2
```

## 8. Document

Add entry to [BASELINES.md](BASELINES.md) with:
- Paper citation
- Difficulty signal
- Trainer kind
- Fair-comparison notes (same epochs, val split, imbalance method)

## Fair comparison checklist

- [ ] Same total epoch budget as `is_cl`
- [ ] Same `max_length`, `batch_size`, `lr`
- [ ] Same `imbalance_method`
- [ ] Record extra compute (e.g. nuclear norm time) in `timings.csv`

## Paper fidelity checklist (SPDCL exemplar)

Use when validating a literature baseline against its source paper:

1. **Algorithm mapping** — document each paper step → code function (see [BASELINES.md](BASELINES.md) b2 table).
2. **Hyperparameter profile** — dedicated YAML with paper-near values (`experiments/spdcl_paper_near.yaml`).
3. **Intentional adaptations** — list model (RoBERTa vs BERT), datasets, val split, extra losses.
4. **Signal timing** — e.g. pretrain norms computed before first gradient step (`NuclearNormScorer.score_pretrain`).
5. **Schedule semantics** — curriculum epochs vs anneal epochs explicitly separated.
6. **Skip unrelated pipeline stages** — e.g. BIOIS not run for SPDCL.
7. **Integration test** — ModernBERT smoke on small dataset (`tests/integration/test_b2_modernbert_smoke.py`).
8. **Timing artifacts** — log method-specific costs (`nuclear_norm_time_s`).
